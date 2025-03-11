import numpy as np
import pandas as pd

# Load the .npy file
file_path = './Image_Train_Model/processed_train_metadata.npy'
data = np.load(file_path, allow_pickle=True)

# Display data details
print("Data Type:", type(data))
print("Shape:", data.shape)
print("First 5 Rows:\n", data[:5])

# Automatic Data Type Inference
def infer_data_types(data):
    sample_row = data[0]  # Take the first row for inference
    inferred_types = []

    for value in sample_row:
        if isinstance(value, str):
            inferred_types.append('String/ID/Category')
        elif isinstance(value, (int, np.int64, np.int32)):
            inferred_types.append('Integer/Numeric')
        elif isinstance(value, (float, np.float64, np.float32)):
            inferred_types.append('Float/Numeric')
        elif isinstance(value, pd.Timestamp):
            inferred_types.append('Datetime')
        elif isinstance(value, tuple):
            inferred_types.append('Tuple/Geographical Cell')
        else:
            inferred_types.append('Unknown')

    return inferred_types

# Infer data types for each column
column_types = infer_data_types(data)

# Display Results
print("\nInferred Column Types:")
for idx, col_type in enumerate(column_types):
    print(f"Column {idx}: {col_type}")
