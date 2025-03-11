import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import gc  # Garbage Collection
import logging

# Setup logging for detailed tracking
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")

# Data type mapping
dtype_mapping = {
    'id': 'Int64',
    'latitude': 'float64',
    'longitude': 'float64',
    'thumb_original_url': 'string',
    'country': 'string',
    'sequence': 'string',
    'captured_at': 'int64',  # Unix epoch time in milliseconds
    'lon_bin': 'int64',
    'lat_bin': 'int64',
    'cell': 'string',
    'region': 'string',
    'sub-region': 'string',
    'city': 'string',
    'land_cover': 'int64',
    'road_index': 'float64',
    'drive_side': 'int64',
    'climate': 'int64',
    'soil': 'int64',
    'dist_sea': 'float64',
    'quadtree_10_5000': 'int64',
    'quadtree_10_25000': 'int64',
    'quadtree_10_1000': 'int64',
    'quadtree_10_50000': 'int64',
    'quadtree_10_12500': 'int64',
    'quadtree_10_500': 'int64',
    'quadtree_10_2500': 'int64',
    'unique_region': 'string',
    'unique_sub-region': 'string',
    'unique_city': 'string',
    'unique_country': 'string',
    'creator_username': 'string',
    'creator_id': 'string'
}

# Function to read data in chunks
def load_csv_from_s3_in_batches(s3_path, chunk_size=100000):
    for chunk in pd.read_csv(s3_path, 
                             chunksize=chunk_size, 
                             dtype=dtype_mapping, 
                             low_memory=False,
                             na_values=["", " ", "NA", "NULL"]):
        yield chunk
        

# Improved Batch Processing Function
def process_batch(batch):
    logging.info("Processing batch of size: %s", batch.shape)

    # Drop unnecessary column
    batch.drop(columns=['thumb_original_url'], errors='ignore', inplace=True)

    # Correct 'id' and 'creator_id' column types
    batch['id'] = batch['id'].astype(str)
    batch['creator_id'] = batch['creator_id'].astype(str)

    # Trim Whitespace in Categorical Columns
    cat_cols = ['region', 'sub-region', 'city', 'unique_country', 'country']
    batch[cat_cols] = batch[cat_cols].apply(lambda x: x.str.strip())

    # Convert 'captured_at' safely to datetime
    batch['captured_at'] = pd.to_datetime(batch['captured_at'], unit='ms', errors='coerce')
    batch['year'] = batch['captured_at'].dt.year.fillna(1970).astype(int)
    batch['month'] = batch['captured_at'].dt.month.fillna(1).astype(int)
    batch['day'] = batch['captured_at'].dt.day.fillna(1).astype(int)

    # Forward Fill and Backward Fill for Categorical Columns
    ffill_cols = ['region', 'sub-region', 'city', 'unique_country', 'unique_region', 'unique_sub-region', 'unique_city']
    for col in ffill_cols:
        batch.loc[:, col] = batch[col].ffill().bfill()

    # Force replacement for remaining gaps in `unique_city` and `creator_username`
    batch['unique_city'].fillna("Unknown", inplace=True)
    batch['creator_username'].fillna("Unknown", inplace=True)

    # Logging - Highlight missing value patterns
    missing_data = batch.isna().sum()
    if missing_data.any():
        logging.warning("❗ Remaining missing values in batch:\n%s", missing_data[missing_data > 0])
        
        # Log specific rows with missing data
        for col in missing_data[missing_data > 0].index:
            missing_rows = batch[batch[col].isna()]
            logging.warning(f"🔍 Rows with missing `{col}`:\n{missing_rows}")

    # Encoding Categorical Variables
    categorical_cols = ['country', 'region', 'sub-region', 'city', 'unique_country']
    for col in categorical_cols:
        le = LabelEncoder()
        batch[col] = le.fit_transform(batch[col].astype(str))

    # Fill Numeric Missing Values
    numeric_cols = batch.select_dtypes(include=['float64', 'int64']).columns
    batch[numeric_cols] = batch[numeric_cols].fillna(0)

    # Normalize Continuous Variables
    continuous_cols = ['latitude', 'longitude', 'dist_sea', 'road_index']
    scaler = StandardScaler()
    batch[continuous_cols] = scaler.fit_transform(batch[continuous_cols])

    # Final Check
    missing_data = batch.isna().sum()
    if missing_data.any():
        logging.warning("❗ Remaining missing values in batch:\n%s", missing_data[missing_data > 0])
    else:
        logging.info("✅ No missing values in batch after cleaning")

    return batch

# Process CSV in Batches
s3_path = 's3://imagescalocation/images/train001.csv'
processed_data = []  # Store processed data for concatenation

for batch in load_csv_from_s3_in_batches(s3_path):
    processed_batch = process_batch(batch)
    processed_data.append(processed_batch)

    # Garbage Collection to free up memory
    del batch
    del processed_batch
    gc.collect()

# Combine all processed data
final_test_data = pd.concat(processed_data, ignore_index=True)

# Final Check Before Saving
total_missing = final_test_data.isna().sum().sum()
logging.info("Total missing values in final dataset before saving: %d", total_missing)
assert total_missing == 0, "❌ There are still missing values in the dataset!"

# Save processed metadata for model input
np.save('processed_train_metadata.npy', final_test_data.values)
logging.info("✅ Saved metadata as `processed_train_metadata.npy`")

# Verification After Loading
loaded_data = np.load('processed_train_metadata.npy', allow_pickle=True)
logging.info("✅ Successfully loaded data. Shape: %s, Dtype: %s", loaded_data.shape, loaded_data.dtype)

# Final Sanity Check on Loaded Data
if loaded_data.dtype.kind == 'f':
    logging.info("✅ Any NaN in loaded data: %s", np.isnan(loaded_data).any())
else:
    logging.info("✅ No NaN detected (integer dtype ensures no NaNs)")
