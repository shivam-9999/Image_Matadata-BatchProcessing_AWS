import boto3
import zipfile
import io
from PIL import Image
import torch
from torchvision import transforms
import gc  # Import garbage collector

# Initialize Boto3 client
s3 = boto3.client('s3')

# Custom Transform for Dynamic Resizing + Padding
class ResizeWithPadding:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def __call__(self, img):
        img.thumbnail(self.target_size)
        padded_img = Image.new("RGB", self.target_size, (0, 0, 0))
        padded_img.paste(img, ((self.target_size[0] - img.width) // 2,
                               (self.target_size[1] - img.height) // 2))
        return padded_img

# Transform Pipeline for Image Processing
transform = transforms.Compose([
    ResizeWithPadding(target_size=(224, 224)),
    transforms.ToTensor()  # Ensures all images are tensors with consistent size
])

# Initialize global batch counter
global_batch_count = 768  # Start from batch 768 onwards

# Function to read ZIP files efficiently from S3 in chunks
def read_images_from_s3_zip(s3_path):
    global global_batch_count  # Maintain global batch count
    print(f"📂 Reading ZIP file: {s3_path}")
    images = []
    obj = s3.get_object(Bucket='imagescalocation', Key=s3_path.replace('s3://imagescalocation/', ''))

    with zipfile.ZipFile(io.BytesIO(obj['Body'].read())) as zip_ref:
        for idx, file_name in enumerate(zip_ref.namelist()):
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                with zip_ref.open(file_name) as file:
                    try:
                        image = Image.open(io.BytesIO(file.read())).convert('RGB')
                        transformed_image = transform(image)
                        images.append((file_name, transformed_image))

                        if len(images) % 200 == 0:  # Save every 200 images
                            global_batch_count += 1
                            print(f"✅ Saving batch {global_batch_count} with {len(images)} images")
                            save_images_to_s3(images,
                                              f'Processed_Train_Images/processed_train_images_batch_{global_batch_count}.pt')
                            del images
                            gc.collect()
                            images = []
                    except Exception as e:
                        print(f"Error loading {file_name}: {e}")

    if images:
        global_batch_count += 1
        print(f"✅ Saving final batch {global_batch_count} with {len(images)} images")
        save_images_to_s3(images,
                          f'Processed_Train_Images/processed_train_images_batch_{global_batch_count}.pt')

    print(f"✅ Processed and saved {global_batch_count} batches successfully!")

    # Remove ZIP file from S3 after processing
    zip_key = s3_path.replace('s3://imagescalocation/', '')
    s3.delete_object(Bucket='imagescalocation', Key=zip_key)
    print(f"🗑️ Deleted ZIP file from S3: {zip_key}")

# Save processed image data and upload to S3
def save_images_to_s3(images, s3_path):
    image_tensors = torch.stack([img for _, img in images])
    buffer = io.BytesIO()
    torch.save(image_tensors, buffer)
    buffer.seek(0)

    s3.upload_fileobj(buffer, 'imagescalocation', s3_path)
    print(f"✅ Uploaded {len(images)} images to {s3_path}")

# Process Train Parts
def process_train_parts():
    for part_num in [f'{i:03}' for i in range(17, 18) if i != 9]:  # Excluding part 008, starting from part_001
        read_images_from_s3_zip(f's3://imagescalocation/images/train/part_{part_num}.zip')
        gc.collect()

# Execute Processing
if __name__ == "__main__":
    process_train_parts()
    print("✅ All Train Parts Processed and Uploaded Successfully!")
