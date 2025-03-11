import torch
import ssl
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

ssl._create_default_https_context = ssl._create_unverified_context

# -------------------- Dataset Class --------------------
class ImageMetadataDataset(Dataset):
    def __init__(self, image_file, metadata_file):
        self.images = torch.load(image_file)
        self.metadata = np.load(metadata_file, allow_pickle=True)

        # Extract labels for latitude and longitude
        self.labels = self.metadata[:, 1:3].astype(np.float32)

        # Replace '<NA>' with 0 and scale metadata
        self.metadata_features = np.where(self.metadata[:, [-4, -3, -2, -1]] == '<NA>', 0, self.metadata[:, [-4, -3, -2, -1]])
        self.metadata_features = MinMaxScaler().fit_transform(self.metadata_features.astype(np.float32))

        # Data Augmentation
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomCrop(224, pad_if_needed=True),
            transforms.GaussianBlur(kernel_size=(3, 3)),
            transforms.RandomErasing(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = self.transform(image)
        metadata = torch.tensor(self.metadata_features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, metadata, label

# -------------------- Improved CNN Model --------------------
class GeolocationCNN(nn.Module):
    def __init__(self):
        super(GeolocationCNN, self).__init__()
        self.cnn = models.efficientnet_b0(pretrained=True)
        self.cnn.classifier[1] = nn.Sequential(
            nn.Linear(self.cnn.classifier[1].in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 2)  # Predict latitude & longitude
        )

    def forward(self, x):
        return self.cnn(x)

# -------------------- Train Function --------------------
def train_model(model, train_loader, val_loader, epochs=30):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for images, metadata, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss = evaluate_model(model, val_loader)
        scheduler.step(val_loss)

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {total_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

# -------------------- Evaluation Function --------------------
def evaluate_model(model, test_loader):
    model.eval()
    total_distance = 0.0
    with torch.no_grad():
        for images, metadata, labels in test_loader:
            outputs = model(images)
            total_distance += torch.mean(torch.sqrt(torch.sum((outputs - labels) ** 2, dim=1))).item()

    return total_distance / len(test_loader)

# -------------------- Main Execution --------------------
if __name__ == "__main__":
    # Dataset Paths
    train_image_file = './Processed_Images/processed_train_images.pt'
    train_metadata_file = './Processed_Files/processed_train_metadata.npy'
    test_image_file = './Processed_Images/processed_test_images.pt'
    test_metadata_file = './Processed_Files/processed_test_metadata.npy'

    # Datasets and Loaders
    full_dataset = ImageMetadataDataset(train_image_file, train_metadata_file)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Model Initialization
    model = GeolocationCNN().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Train and Evaluate
    train_model(model, train_loader, val_loader, epochs=30)
    final_distance = evaluate_model(model, test_loader)
    print(f"🏁 Final Model Evaluation (Haversine Distance): {final_distance:.4f} km")

    # Save Model
    torch.save(model.state_dict(), 'geolocation_cnn_improved.pth')
    print("✅ Model training complete and saved successfully!")