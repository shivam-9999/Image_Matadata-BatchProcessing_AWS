import boto3
import torch
import io
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# Initialize Boto3 client
s3 = boto3.client('s3')

# Function to Load Processed Images from S3
def load_images_from_s3(s3_bucket, s3_key):
    obj = s3.get_object(Bucket=s3_bucket, Key=s3_key)
    image_tensors = torch.load(io.BytesIO(obj['Body'].read()), map_location='cpu')
    return image_tensors

# Load Processed Images
bucket_name = 'imagescalocation'
test_s3_path = 'Processed_Test_Images/processed_test_images_batch_1.pt'  # ✅ Corrected Path

# Load processed images directly from S3
image_tensors = load_images_from_s3(bucket_name, test_s3_path)

# Display shape information
print(f"Total Images: {image_tensors.shape[0]}")
print(f"Sample Image Shape: {image_tensors[0].shape}")

# Inverse Transform (Tensor → Image)
inverse_transform = transforms.ToPILImage()

# Display Sample Images
def show_images(images, num_samples=5):
    plt.figure(figsize=(12, 6))
    for i in range(min(num_samples, len(images))):
        img = inverse_transform(images[i])  # Convert Tensor back to PIL
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img)
        plt.axis('off')
    plt.show()

# Display a subset of images (e.g., first 5 images)
show_images(image_tensors[180:185])
