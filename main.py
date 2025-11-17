from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import base64
from typing import Optional


# ============================================
# MODELO
# ============================================

def create_transfer_learning_model():
    """Crea el modelo ResNet18 con capas personalizadas"""
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )
    return model


# ============================================
# TRANSFORMACIONES
# ============================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ============================================
# PYDANTIC MODELS
# ============================================

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict
    emoji: str


# ============================================
# FASTAPI APP
# ============================================

app = FastAPI(
    title="🐱🐶 Cat vs Dog Classifier API",
    description="API para clasificar imágenes de gatos y perros usando Deep Learning",
    version="1.0.0"
)

# CORS para permitir peticiones desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# CARGAR MODELO AL INICIAR
# ============================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None


@app.on_event("startup")
async def load_model():
    """Carga el modelo al iniciar la aplicación"""
    global model
    try:
        model = create_transfer_learning_model().to(device)
        model.load_state_dict(torch.load('best_accuracy_model.pth', map_location=device))
        model.eval()
        print(f"✅ Modelo cargado exitosamente en {device}")
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        raise


# ============================================
# FUNCIÓN DE PREDICCIÓN
# ============================================

def predict_image(image: Image.Image) -> dict:
    """Realiza la predicción sobre una imagen"""

    # Preprocesar imagen
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predicción
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    # Resultados
    class_names = ['Gato', 'Perro']
    emojis = ['🐱', '🐶']

    return {
        'prediction': class_names[predicted_class],
        'confidence': round(confidence * 100, 2),
        'probabilities': {
            'cat': round(probabilities[0][0].item() * 100, 2),
            'dog': round(probabilities[0][1].item() * 100, 2)
        },
        'emoji': emojis[predicted_class]
    }


# ============================================
# ENDPOINTS
# ============================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Página principal con interfaz HTML"""
    html_content = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🐱🐶 Cat vs Dog Classifier</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            .container {
                background: white;
                padding: 40px;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                max-width: 600px;
                width: 100%;
            }
            h1 {
                text-align: center;
                color: #333;
                margin-bottom: 10px;
                font-size: 2.5em;
            }
            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 30px;
            }
            .upload-area {
                border: 3px dashed #667eea;
                border-radius: 15px;
                padding: 40px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s;
                margin-bottom: 20px;
            }
            .upload-area:hover {
                border-color: #764ba2;
                background: #f8f9ff;
            }
            .upload-area.dragover {
                background: #e8ebff;
                border-color: #764ba2;
            }
            input[type="file"] { display: none; }
            .upload-icon {
                font-size: 4em;
                margin-bottom: 10px;
            }
            .upload-text {
                color: #666;
                font-size: 1.1em;
            }
            #preview {
                max-width: 100%;
                max-height: 400px;
                border-radius: 10px;
                margin: 20px auto;
                display: none;
            }
            #result {
                margin-top: 30px;
                padding: 20px;
                border-radius: 10px;
                display: none;
                text-align: center;
            }
            #result.cat {
                background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
            }
            #result.dog {
                background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
            }
            .result-emoji {
                font-size: 5em;
                margin-bottom: 10px;
            }
            .result-text {
                font-size: 2em;
                font-weight: bold;
                color: #2d3436;
                margin-bottom: 10px;
            }
            .confidence {
                font-size: 1.3em;
                color: #2d3436;
            }
            .probabilities {
                display: flex;
                justify-content: space-around;
                margin-top: 20px;
                gap: 20px;
            }
            .prob-item {
                flex: 1;
                background: rgba(255,255,255,0.5);
                padding: 15px;
                border-radius: 10px;
            }
            .prob-label {
                font-size: 1.2em;
                margin-bottom: 5px;
            }
            .prob-value {
                font-size: 1.5em;
                font-weight: bold;
            }
            .loading {
                display: none;
                text-align: center;
                margin: 20px 0;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .btn-reset {
                background: #667eea;
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 8px;
                font-size: 1.1em;
                cursor: pointer;
                margin-top: 20px;
                transition: all 0.3s;
            }
            .btn-reset:hover {
                background: #764ba2;
                transform: translateY(-2px);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🐱🐶 Cat vs Dog</h1>
            <p class="subtitle">Clasificador de imágenes con Deep Learning</p>

            <div class="upload-area" id="uploadArea">
                <div class="upload-icon">📸</div>
                <div class="upload-text">
                    Haz clic o arrastra una imagen aquí
                </div>
                <input type="file" id="fileInput" accept="image/*">
            </div>

            <img id="preview" alt="Preview">

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p style="margin-top: 10px; color: #666;">Analizando imagen...</p>
            </div>

            <div id="result">
                <div class="result-emoji" id="resultEmoji"></div>
                <div class="result-text" id="resultText"></div>
                <div class="confidence" id="confidence"></div>
                <div class="probabilities">
                    <div class="prob-item">
                        <div class="prob-label">🐱 Gato</div>
                        <div class="prob-value" id="catProb"></div>
                    </div>
                    <div class="prob-item">
                        <div class="prob-label">🐶 Perro</div>
                        <div class="prob-value" id="dogProb"></div>
                    </div>
                </div>
                <button class="btn-reset" onclick="reset()">Analizar otra imagen</button>
            </div>
        </div>

        <script>
            const uploadArea = document.getElementById('uploadArea');
            const fileInput = document.getElementById('fileInput');
            const preview = document.getElementById('preview');
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');

            // Click para seleccionar archivo
            uploadArea.addEventListener('click', () => fileInput.click());

            // Drag and drop
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });

            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });

            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                const file = e.dataTransfer.files[0];
                if (file && file.type.startsWith('image/')) {
                    handleFile(file);
                }
            });

            // Selección de archivo
            fileInput.addEventListener('change', (e) => {
                const file = e.target.files[0];
                if (file) handleFile(file);
            });

            async function handleFile(file) {
                // Mostrar preview
                const reader = new FileReader();
                reader.onload = (e) => {
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                };
                reader.readAsDataURL(file);

                // Ocultar área de upload y resultado anterior
                uploadArea.style.display = 'none';
                result.style.display = 'none';
                loading.style.display = 'block';

                // Enviar al servidor
                const formData = new FormData();
                formData.append('file', file);

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();

                    // Mostrar resultado
                    loading.style.display = 'none';
                    result.style.display = 'block';
                    result.className = data.prediction.toLowerCase();

                    document.getElementById('resultEmoji').textContent = data.emoji;
                    document.getElementById('resultText').textContent = data.prediction;
                    document.getElementById('confidence').textContent = 
                        `Confianza: ${data.confidence}%`;
                    document.getElementById('catProb').textContent = 
                        `${data.probabilities.cat}%`;
                    document.getElementById('dogProb').textContent = 
                        `${data.probabilities.dog}%`;

                } catch (error) {
                    loading.style.display = 'none';
                    alert('Error al procesar la imagen: ' + error);
                }
            }

            function reset() {
                uploadArea.style.display = 'block';
                preview.style.display = 'none';
                result.style.display = 'none';
                fileInput.value = '';
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Endpoint para realizar predicciones"""

    # Validar tipo de archivo
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    try:
        # Leer imagen
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        # Predecir
        result = predict_image(image)

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")


@app.get("/health")
async def health_check():
    """Endpoint para verificar el estado de la API"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }


@app.get("/info")
async def model_info():
    """Información sobre el modelo"""
    return {
        "model": "ResNet18 Transfer Learning",
        "classes": ["Gato", "Perro"],
        "input_size": "224x224",
        "device": str(device),
        "model_file": "best_accuracy_model.pth"
    }