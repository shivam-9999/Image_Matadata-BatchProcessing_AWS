import boto3
import os
import pandas as pd
from tqdm import tqdm

# AWS Configuration
AWS_REGION = 'ca-central-1'  # Change this as per your AWS region
S3_BUCKET_NAME = 'imagescalocation'
S3_TRAIN_FOLDER = 'images/train/'  # Destination folder inside your S3 bucket
LOCAL_CSV_PATH = 'data/large_file.csv'  # Path to your large CSV file
SPLIT_FOLDER = 'data/split_files/'  # Local folder for split files

# Create directories if they don't exist
os.makedirs(SPLIT_FOLDER, exist_ok=True)

# Create an S3 client
s3_client = boto3.client('s3')

# Function to upload a single file with progress tracking
def upload_file_with_progress(file_path, bucket, object_name):
    file_size = os.path.getsize(file_path)
    
    with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Uploading {os.path.basename(file_path)}") as progress_bar:
        def progress_callback(bytes_transferred):
            progress_bar.update(bytes_transferred)

        s3_client.upload_file(file_path, bucket, object_name, Callback=progress_callback)

# Function to split CSV file into 1 GB chunks
def split_csv_file():
    print("Splitting CSV file...")

    chunk_size = 10 ** 6  # Approximate rows per 1GB (adjust based on your data)

    for i, chunk in enumerate(pd.read_csv(LOCAL_CSV_PATH, chunksize=chunk_size)):
        split_file_name = f"split_{i + 1}.csv"
        split_file_path = os.path.join(SPLIT_FOLDER, split_file_name)
        
        chunk.to_csv(split_file_path, index=False)
        print(f"✅ {split_file_name} created successfully")

# Function to upload split CSV files sequentially
def upload_split_files():
    csv_files = [f for f in os.listdir(SPLIT_FOLDER) if f.endswith('.csv')]

    if not csv_files:
        print("No split files found in the split folder.")
        return

    for csv_file in csv_files:
        csv_path = os.path.join(SPLIT_FOLDER, csv_file)
        s3_object_name = os.path.join(S3_TRAIN_FOLDER, csv_file)

        print(f"\nStarting upload for: {csv_file}")
        upload_file_with_progress(csv_path, S3_BUCKET_NAME, s3_object_name)
        print(f"✅ {csv_file} uploaded successfully!")

# Run the process
if __name__ == "__main__":
    split_csv_file()
    upload_split_files()
