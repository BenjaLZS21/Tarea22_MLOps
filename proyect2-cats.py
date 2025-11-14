import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os


# 1. Definir el modelo CNN
class CatDogClassifier(nn.Module):
    def __init__(self):
        super(CatDogClassifier, self).__init__()

        # Capas convolucionales
        self.conv_layers = nn.Sequential(
            # Primera capa convolucional
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Segunda capa convolucional
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Tercera capa convolucional
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Capas fully connected
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)  # 2 clases: gato y perro
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# 2. Dataset personalizado
class CatDogDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir: Directorio con subdirectorios 'cats' y 'dogs'
            transform: Transformaciones a aplicar a las imágenes
        """
        self.image_dir = image_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # Cargar imágenes de gatos (label 0)
        cats_dir = os.path.join(image_dir, 'cats')
        if os.path.exists(cats_dir):
            for img_name in os.listdir(cats_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(cats_dir, img_name))
                    self.labels.append(0)

        # Cargar imágenes de perros (label 1)
        dogs_dir = os.path.join(image_dir, 'dogs')
        if os.path.exists(dogs_dir):
            for img_name in os.listdir(dogs_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(dogs_dir, img_name))
                    self.labels.append(1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# 3. Transformaciones de datos
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# 4. Función de entrenamiento
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass y optimización
            loss.backward()
            optimizer.step()

            # Estadísticas
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}] completada - '
              f'Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%\n')


# 5. Función de evaluación
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy en el conjunto de prueba: {accuracy:.2f}%')
    return accuracy


# 6. Función de predicción para una imagen individual
def predict_image(model, image_path, transform, device):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    class_names = ['Gato', 'Perro']
    return class_names[predicted_class], confidence


# 7. Función principal
def main():
    # Configuración
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Usando dispositivo: {device}')

    # Crear datasets y dataloaders
    # NOTA: Debes tener tus imágenes organizadas así:
    # data/
    #   train/
    #     cats/
    #     dogs/
    #   test/
    #     cats/
    #     dogs/

    train_dataset = CatDogDataset('data/train', transform=train_transform)
    test_dataset = CatDogDataset('data/test', transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Crear modelo
    model = CatDogClassifier().to(device)

    # Pérdida y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Entrenar
    print("Iniciando entrenamiento...")
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=10)

    # Evaluar
    print("\nEvaluando modelo...")
    evaluate_model(model, test_loader, device)

    # Guardar modelo
    torch.save(model.state_dict(), 'cat_dog_model1.pth')
    print("Modelo guardado como 'cat_dog_model1.pth'")

    # Ejemplo de predicción
    # prediction, confidence = predict_image(model, 'imagen_test.jpg', test_transform, device)
    # print(f'Predicción: {prediction} (Confianza: {confidence:.2%})')


if __name__ == '__main__':
    main()