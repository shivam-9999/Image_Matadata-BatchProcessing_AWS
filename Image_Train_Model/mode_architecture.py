import os
import boto3
import torch
import ssl
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models import EfficientNet_B0_Weights
import itertools  # For cycling through test files

ssl._create_default_https_context = ssl._create_unverified_context

# -------------------- S3 Utility --------------------
s3_client = boto3.client('s3')
bucket_name = 'imagescalocation'

def list_s3_files(prefix):
    """Dynamically list .pt files in S3"""
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    return sorted([obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.pt')])

def load_pt_file(bucket_name, key):
    """Load .pt file from S3 using io.BytesIO for seekable buffer"""
    obj = s3_client.get_object(Bucket=bucket_name, Key=key)
    buffer = io.BytesIO(obj['Body'].read())
    return torch.load(buffer)

# -------------------- Dataset Class --------------------
class ImageMetadataDataset(Dataset):
    def __init__(self, image_files, metadata, labels, metadata_features):
        self.images = []
        for file in image_files:
            print(f"📂 Reading {file}")
            images = load_pt_file(bucket_name, file)
            self.images.extend(images)

        self.metadata = metadata
        self.labels = labels
        self.metadata_features = metadata_features

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
        self.cnn = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
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
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

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

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {total_loss / len(train_loader):.4f}, "
              f"Val Loss: {val_loss:.4f}")

def evaluate_model(model, test_loader):
    model.eval()
    total_distance = 0.0
    with torch.no_grad():
        for images, metadata, labels in test_loader:
            outputs = model(images)
            total_distance += torch.mean(torch.sqrt(torch.sum((outputs - labels) ** 2, dim=1))).item()

    return total_distance / len(test_loader)

# -------------------- Sequential Training Logic --------------------
if __name__ == "__main__":
    train_files = list_s3_files('Processed_Train_Images/')
    test_files = list_s3_files('Processed_Test_Images/')

    train_metadata = np.load(os.environ['SM_CHANNEL_TRAIN'] + '/processed_train_metadata.npy', allow_pickle=True)
    test_metadata = np.load(os.environ['SM_CHANNEL_TEST'] + '/processed_test_metadata.npy', allow_pickle=True)

    train_labels = train_metadata[:, 1:3].astype(np.float32)
    train_metadata_features = MinMaxScaler().fit_transform(train_metadata[:, [-4, -3, -2, -1]].astype(np.float32))

    test_labels = test_metadata[:, 1:3].astype(np.float32)
    test_metadata_features = MinMaxScaler().fit_transform(test_metadata[:, [-4, -3, -2, -1]].astype(np.float32))

    # Model Initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeolocationCNN().to(device)

    # Dynamic Iteration (More Train Files Than Test Files)
    test_index = 0
    for train_file in train_files:
        print(f"\n🟦 Reading {train_file} & {test_files[test_index]}")

        train_dataset = ImageMetadataDataset([train_file], train_metadata, train_labels, train_metadata_features)
        test_dataset = ImageMetadataDataset([test_files[test_index]], test_metadata, test_labels, test_metadata_features)

        # Cycle through test files dynamically
        test_index = (test_index + 1) % len(test_files)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=4)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=4)

        # Train and Evaluate
        train_model(model, train_loader, test_loader, epochs=10)
        final_distance = evaluate_model(model, test_loader)
        print(f"🏁 Evaluation (Haversine Distance): {final_distance:.4f} km")

        # Garbage Collection
        del train_dataset, test_dataset
        torch.cuda.empty_cache()
