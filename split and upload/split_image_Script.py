import os
import pandas as pd
import csv

# Constants
SOURCE_CSV = "images/train.csv"
OUTPUT_DIR = "output/train/"
CHUNK_SIZE = 1 * 1024 * 1024 * 1024  # 1GB limit

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_next_chunk_number(output_dir):
    """Find the next available chunk number (e.g., part_004.csv)."""
    existing_chunks = [
        int(filename.split('_')[1].split('.')[0])
        for filename in os.listdir(output_dir)
        if filename.startswith('part_') and filename.endswith('.csv')
    ]
    return max(existing_chunks, default=0) + 1

def split_csv_file(source_csv, output_dir, chunk_size):
    """Splits a large CSV file into smaller CSV files (~1GB each)."""
    next_chunk_index = get_next_chunk_number(output_dir)

    # Read CSV in chunks to avoid memory overload
    total_bytes = 0
    chunk_data = []

    with open(source_csv, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Read the header row

        for row in reader:
            row_size = sum(len(str(value)) for value in row) + 1  # Approximate row size
            total_bytes += row_size
            chunk_data.append(row)

            if total_bytes >= chunk_size:
                output_file = os.path.join(output_dir, f"train{next_chunk_index:03d}.csv")
                write_chunk(output_file, headers, chunk_data)

                print(f"✅ Created: {output_file}")
                next_chunk_index += 1
                total_bytes = 0
                chunk_data = []

        # Write remaining data (if any)
        if chunk_data:
            output_file = os.path.join(output_dir, f"part_{next_chunk_index:03d}.csv")
            write_chunk(output_file, headers, chunk_data)
            print(f"✅ Created: {output_file}")

    print(f"\n✅ Splitting complete! New CSV files created in '{output_dir}'")

def write_chunk(output_file, headers, data):
    """Writes data to a CSV file."""
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)

# Run the script
if __name__ == "__main__":
    if not os.path.exists(SOURCE_CSV):
        print(f"❌ File not found: {SOURCE_CSV}")
    else:
        split_csv_file(SOURCE_CSV, OUTPUT_DIR, CHUNK_SIZE)
