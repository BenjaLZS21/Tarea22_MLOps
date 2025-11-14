import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import copy


# Dataset personalizado (igual que antes)
class CatDogDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = []
        self.labels = []

        cats_dir = os.path.join(image_dir, 'cats')
        if os.path.exists(cats_dir):
            for img_name in os.listdir(cats_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(cats_dir, img_name))
                    self.labels.append(0)

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


# Transformaciones mejoradas con más augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
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


# Early Stopping
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)


# Modelo con Transfer Learning (ResNet18)
def create_transfer_learning_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Congelar las primeras capas
    for param in model.parameters():
        param.requires_grad = False

    # Descongelar las últimas capas
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Reemplazar la capa final
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


# Función de entrenamiento mejorada con validación
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=20):
    early_stopping = EarlyStopping(patience=5, path='best_cat_dog_model.pth')
    best_acc = 0.0

    for epoch in range(num_epochs):
        # Fase de entrenamiento
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # Fase de validación
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total

        print(f'\nEpoch [{epoch + 1}/{num_epochs}]:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n')

        # Ajustar learning rate
        scheduler.step(val_loss)

        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping activado")
            break

        # Guardar mejor modelo
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_accuracy_model.pth')
            print(f'Nuevo mejor modelo guardado con {val_acc:.2f}% accuracy')


# Función de evaluación
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    class_correct = [0, 0]
    class_total = [0, 0]

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Precisión por clase
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    accuracy = 100 * correct / total
    print(f'\nAccuracy total: {accuracy:.2f}%')
    print(f'Accuracy Gatos: {100 * class_correct[0] / class_total[0]:.2f}%')
    print(f'Accuracy Perros: {100 * class_correct[1] / class_total[1]:.2f}%')
    return accuracy


# Función principal
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Usando dispositivo: {device}')

    # Crear datasets
    train_dataset = CatDogDataset('data/train', transform=train_transform)
    test_dataset = CatDogDataset('data/test', transform=test_transform)

    # Dividir train en train y validación (80-20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    print(f'Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}')

    # Crear modelo con Transfer Learning
    model = create_transfer_learning_model().to(device)

    # Optimizador y scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # Entrenar
    print("\nIniciando entrenamiento con Transfer Learning...")
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=20)

    # Cargar mejor modelo y evaluar
    print("\nCargando mejor modelo...")
    model.load_state_dict(torch.load('best_accuracy_model.pth'))
    evaluate_model(model, test_loader, device)


if __name__ == '__main__':
    main()