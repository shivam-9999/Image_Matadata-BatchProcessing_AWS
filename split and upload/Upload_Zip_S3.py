import boto3
import os
from tqdm import tqdm

# AWS Configuration
AWS_REGION = 'ca-central-1'  # Change this as per your AWS region
S3_BUCKET_NAME = 'imagescalocation/images/'  # Change this to your bucket name
S3_TRAIN_FOLDER = 'train/'  # Destination folder inside your S3 bucket
LOCAL_TRAIN_FOLDER = 'images/train/'  # Local folder where ZIP files are stored

# Create an S3 client
s3_client = boto3.client('s3')

# Function to upload a single file with progress tracking
def upload_file_with_progress(file_path, bucket, object_name):
    file_size = os.path.getsize(file_path)
    
    # Progress bar setup
    with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Uploading {os.path.basename(file_path)}") as progress_bar:
        def progress_callback(bytes_transferred):
            progress_bar.update(bytes_transferred)

        # Upload file with progress callback
        s3_client.upload_file(file_path, bucket, object_name, Callback=progress_callback)

# Function to upload ZIP files sequentially (one at a time)
def upload_zip_files():
    zip_files = [f for f in os.listdir(LOCAL_TRAIN_FOLDER) if f.endswith('.zip')]
    
    if not zip_files:
        print("No ZIP files found in the train folder.")
        return

    for zip_file in zip_files: 
        zip_path = os.path.join(LOCAL_TRAIN_FOLDER, zip_file)
        s3_object_name = os.path.join(S3_TRAIN_FOLDER, zip_file)

        print(f"\nStarting upload for: {zip_file}")
        upload_file_with_progress(zip_path, S3_BUCKET_NAME, s3_object_name)
        print(f"✅ {zip_file} uploaded successfully!")

# Run the upload process
if __name__ == "__main__":
    upload_zip_files()
