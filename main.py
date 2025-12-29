from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import trimesh
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import io
import base64

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "online", "message": "Conversor 3D Operante"}

@app.get("/health")
def health_check():
    return {"status": "ok"}



# Permite que qualquer site (Lovable, etc) acesse sua API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def process_glb_logic(file_bytes, n_colors):
    # (A MESMA LÓGICA DE ANTES, SÓ ADAPTADA PARA RETORNAR DADOS PUROS)
    file_obj = io.BytesIO(file_bytes)
    mesh = trimesh.load(file_obj, file_type='glb', force='mesh')
    
    # Detecção de Textura
    texture = None
    try:
        mat = mesh.visual.material
        if isinstance(mat, list): mat = mat[0]
        if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture: texture = mat.baseColorTexture
        elif hasattr(mat, 'image') and mat.image: texture = mat.image
        elif isinstance(mat, dict) and 'image' in mat: texture = mat['image']
    except Exception: pass

    if texture is None: return None, None

    if texture.mode != 'RGB': texture = texture.convert('RGB')
    tex_array = np.array(texture)
    h, w, _ = tex_array.shape

    uvs = mesh.visual.uv
    faces = mesh.faces
    face_uvs = uvs[faces].mean(axis=1)
    
    u = (face_uvs[:, 0] * (w - 1)).astype(int).clip(0, w - 1)
    v = ((1 - face_uvs[:, 1]) * (h - 1)).astype(int).clip(0, h - 1)
    face_colors = tex_array[v, u]

    kmeans = MiniBatchKMeans(n_clusters=n_colors, batch_size=4096, n_init=3)
    labels = kmeans.fit_predict(face_colors)
    centroids = kmeans.cluster_centers_.astype(int)

    sub_meshes = []
    palette_data = []

    for i in range(n_colors):
        mask = labels == i
        if not np.any(mask): continue
        
        part = mesh.submesh([mask], append=True)
        if isinstance(part, list): part = trimesh.util.concatenate(part)
        
        part_name = f"Cor_{i+1}"
        part.metadata['name'] = part_name
        part.visual.face_colors = np.append(centroids[i], 255)
        sub_meshes.append(part)
        
        r, g, b = centroids[i]
        palette_data.append({
            "name": part_name,
            "rgb": f"{r},{g},{b}",
            "hex": '#{:02x}{:02x}{:02x}'.format(r, g, b)
        })

    scene = trimesh.Scene(sub_meshes)
    export_bytes = scene.export(file_type='3mf')
    return export_bytes, palette_data

@app.post("/convert")
async def convert_file(file: UploadFile = File(...), colors: int = Form(...)):
    content = await file.read()
    
    try:
        processed_file, palette = process_glb_logic(content, colors)
        
        if processed_file is None:
            return {"error": "Sem textura encontrada"}
            
        # Converte o arquivo para Base64 para enviar via JSON
        file_b64 = base64.b64encode(processed_file).decode('utf-8')
        
        return {
            "success": True,
            "palette": palette,
            "file_base64": file_b64,
            "filename": "modelo_convertido.3mf"
        }
    except Exception as e:
        return {"error": str(e)}

# Necessário para o Render rodar
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
