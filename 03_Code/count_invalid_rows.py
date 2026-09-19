import pandas as pd
import numpy as np
from pathlib import Path
#Set folder containing merged CSV files
dataset_folder = Path(
    r"C:\Users\ami25\Desktop\Resear\02_Dataset\MERGED_CSV\MERGED_CSV"
)
#Find all merged CSV files
csv_files = sorted(dataset_folder.glob("Merged*.csv"))
total_rows = 0
invalid_rows = 0
#Check each file for invalid rows
for i, file in enumerate(csv_files, start=1):
    print(f"Checking {i}/{len(csv_files)}: {file.name}")
    df = pd.read_csv(file)
    total_rows += len(df)
    #Find numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    #Convert infinite values into missing values
    df[numeric_cols] = df[numeric_cols].replace(
        [np.inf, -np.inf],
        np.nan
    )
    #Find rows containing invalid values
    invalid_mask = df.isnull().any(axis=1)
    invalid_rows += invalid_mask.sum()
#Calculate valid rows remaining
valid_rows = total_rows - invalid_rows
print("\n================================")
print("INVALID ROW SUMMARY")
print("================================")
print(f"Total rows: {total_rows:,}")
print(f"Rows containing at least one missing/infinite value: {invalid_rows:,}")
print(f"Valid rows remaining: {valid_rows:,}")
print(f"Percentage removed: {(invalid_rows / total_rows) * 100:.6f}%")