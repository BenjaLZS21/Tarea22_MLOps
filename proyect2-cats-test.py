import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path


# Definir el modelo (debe ser idéntico al usado en entrenamiento)
class CatDogClassifier(nn.Module):
    def __init__(self):
        super(CatDogClassifier, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# Transformaciones (deben ser las mismas que en entrenamiento)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def load_model(model_path, device):
    """Carga el modelo entrenado"""
    model = CatDogClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✓ Modelo cargado desde: {model_path}")
    return model


def predict_single_image(model, image_path, device, show_image=True):
    """Predice la clase de una sola imagen"""

    # Cargar y preprocesar imagen
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Hacer predicción
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    class_names = ['🐱 Gato', '🐶 Perro']
    prediction = class_names[predicted_class]

    # Mostrar resultado
    print(f"\n{'=' * 50}")
    print(f"Imagen: {os.path.basename(image_path)}")
    print(f"Predicción: {prediction}")
    print(f"Confianza: {confidence:.2%}")
    print(f"Probabilidades -> Gato: {probabilities[0][0]:.2%}, Perro: {probabilities[0][1]:.2%}")
    print(f"{'=' * 50}\n")

    # Mostrar imagen con predicción
    if show_image:
        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f'{prediction} (Confianza: {confidence:.2%})',
                  fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    return prediction, confidence


def predict_batch_images(model, image_dir, device, max_images=None):
    """Predice múltiples imágenes de un directorio"""

    # Obtener todas las imágenes
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []

    for ext in image_extensions:
        image_files.extend(Path(image_dir).glob(f'*{ext}'))
        image_files.extend(Path(image_dir).glob(f'*{ext.upper()}'))

    if max_images:
        image_files = image_files[:max_images]

    print(f"\n🔍 Analizando {len(image_files)} imágenes...\n")

    results = []
    for img_path in image_files:
        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

            class_names = ['Gato', 'Perro']
            results.append({
                'filename': img_path.name,
                'prediction': class_names[predicted_class],
                'confidence': confidence
            })

            print(f"✓ {img_path.name:30s} -> {class_names[predicted_class]:5s} ({confidence:.2%})")

        except Exception as e:
            print(f"✗ Error con {img_path.name}: {e}")

    return results


def evaluate_test_set(model, test_dir, device):
    """Evalúa el modelo en un conjunto de prueba organizado"""

    cats_dir = os.path.join(test_dir, 'cats')
    dogs_dir = os.path.join(test_dir, 'dogs')

    if not os.path.exists(cats_dir) or not os.path.exists(dogs_dir):
        print("⚠️ Estructura de directorios incorrecta.")
        print(f"Esperado: {test_dir}/cats/ y {test_dir}/dogs/")
        return

    correct = 0
    total = 0

    # Evaluar gatos (clase 0)
    print("\n📊 Evaluando gatos...")
    cat_files = list(Path(cats_dir).glob('*.jpg')) + list(Path(cats_dir).glob('*.png'))
    for img_path in cat_files:
        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image_tensor)
                predicted_class = torch.argmax(output, dim=1).item()

            if predicted_class == 0:  # Gato
                correct += 1
            total += 1
        except Exception as e:
            print(f"Error con {img_path.name}: {e}")

    # Evaluar perros (clase 1)
    print("📊 Evaluando perros...")
    dog_files = list(Path(dogs_dir).glob('*.jpg')) + list(Path(dogs_dir).glob('*.png'))
    for img_path in dog_files:
        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image_tensor)
                predicted_class = torch.argmax(output, dim=1).item()

            if predicted_class == 1:  # Perro
                correct += 1
            total += 1
        except Exception as e:
            print(f"Error con {img_path.name}: {e}")

    accuracy = 100 * correct / total if total > 0 else 0

    print(f"\n{'=' * 50}")
    print(f"📈 RESULTADOS DE EVALUACIÓN")
    print(f"{'=' * 50}")
    print(f"Total de imágenes: {total}")
    print(f"Correctas: {correct}")
    print(f"Incorrectas: {total - correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"{'=' * 50}\n")

    return accuracy


def visualize_predictions(model, image_dir, device, num_images=9):
    """Muestra una grilla de predicciones"""

    image_files = list(Path(image_dir).glob('*.jpg')) + list(Path(image_dir).glob('*.png'))
    image_files = image_files[:num_images]

    rows = int(np.sqrt(num_images))
    cols = int(np.ceil(num_images / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten() if num_images > 1 else [axes]

    for idx, img_path in enumerate(image_files):
        if idx >= num_images:
            break

        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

            class_names = ['🐱 Gato', '🐶 Perro']
            color = 'green' if confidence > 0.7 else 'orange' if confidence > 0.5 else 'red'

            axes[idx].imshow(image)
            axes[idx].axis('off')
            axes[idx].set_title(f'{class_names[predicted_class]}\n{confidence:.1%}',
                                fontsize=12, color=color, fontweight='bold')
        except Exception as e:
            axes[idx].text(0.5, 0.5, f'Error', ha='center', va='center')
            axes[idx].axis('off')

    # Ocultar ejes vacíos
    for idx in range(len(image_files), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()


def main():
    # Configuración
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Usando dispositivo: {device}")

    model_path = 'cat_dog_model.pth'

    # Verificar que existe el modelo
    if not os.path.exists(model_path):
        print(f"❌ No se encontró el modelo en: {model_path}")
        print("Primero debes entrenar el modelo.")
        return

    # Cargar modelo
    model = load_model(model_path, device)

    print("\n" + "=" * 50)
    print("🧪 OPCIONES DE TESTING")
    print("=" * 50)
    print("1. Predecir una sola imagen")
    print("2. Predecir múltiples imágenes de un directorio")
    print("3. Evaluar conjunto de prueba (con labels)")
    print("4. Visualizar predicciones en grilla")
    print("=" * 50)

    option = input("\nElige una opción (1-4): ").strip()

    if option == '1':
        image_path = input("Ruta de la imagen: ").strip()
        if os.path.exists(image_path):
            predict_single_image(model, image_path, device)
        else:
            print(f"❌ No se encontró la imagen: {image_path}")

    elif option == '2':
        dir_path = input("Ruta del directorio: ").strip()
        if os.path.exists(dir_path):
            results = predict_batch_images(model, dir_path, device)
        else:
            print(f"❌ No se encontró el directorio: {dir_path}")

    elif option == '3':
        test_dir = input("Ruta del directorio de prueba (con subdirectorios cats/ y dogs/): ").strip()
        if os.path.exists(test_dir):
            evaluate_test_set(model, test_dir, device)
        else:
            print(f"❌ No se encontró el directorio: {test_dir}")

    elif option == '4':
        dir_path = input("Ruta del directorio: ").strip()
        if os.path.exists(dir_path):
            visualize_predictions(model, dir_path, device)
        else:
            print(f"❌ No se encontró el directorio: {dir_path}")

    else:
        print("❌ Opción no válida")


if __name__ == '__main__':
    main()