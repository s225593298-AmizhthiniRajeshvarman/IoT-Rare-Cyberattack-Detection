import pandas as pd
import numpy as np
from pathlib import Path
#Set folder containing merged CSV files
dataset_folder = Path(
    r"C:\Users\ami25\Desktop\Resear\02_Dataset\MERGED_CSV\MERGED_CSV"
)
#Find all merged CSV files
csv_files = sorted(dataset_folder.glob("Merged*.csv"))
total_missing = {}
total_infinite = {}
total_rows = 0
#Check data quality in each file
for i, file in enumerate(csv_files, start=1):
    print(f"Checking {i}/{len(csv_files)}: {file.name}")
    df = pd.read_csv(file)
    total_rows += len(df)
    #Count missing values
    missing_counts = df.isnull().sum()
    #Check numeric columns for infinite values
    numeric_df = df.select_dtypes(include=[np.number])
    infinite_counts = np.isinf(numeric_df).sum()
    #Add counts from each file
    for column, count in missing_counts.items():
        total_missing[column] = total_missing.get(column, 0) + int(count)
    for column, count in infinite_counts.items():
        total_infinite[column] = total_infinite.get(column, 0) + int(count)
print("\n================================")
print("FULL DATASET DATA QUALITY SUMMARY")
print("================================")
print(f"Total rows checked: {total_rows:,}")
print("\nColumns with missing values:")
missing_found = False
for column, count in total_missing.items():
    if count > 0:
        missing_found = True
        print(f"{column}: {count:,}")
if not missing_found:
    print("No missing values found.")
print("\nColumns with infinite values:")
infinite_found = False
for column, count in total_infinite.items():
    if count > 0:
        infinite_found = True
        print(f"{column}: {count:,}")
if not infinite_found:
    print("No infinite values found.")