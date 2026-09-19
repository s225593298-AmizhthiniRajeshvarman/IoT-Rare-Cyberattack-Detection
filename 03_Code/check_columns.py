import pandas as pd
from pathlib import Path
#Set folder containing merged CSV files
dataset_folder = Path(
    r"C:\Users\ami25\Desktop\Resear\02_Dataset\MERGED_CSV\MERGED_CSV"
)
#Find all merged CSV files
csv_files = sorted(dataset_folder.glob("Merged*.csv"))
reference_columns = None
all_match = True
#Check column names in each file
for i, file in enumerate(csv_files, start=1):
    df = pd.read_csv(file, nrows=1)
    columns = df.columns.tolist()
#Use first file as reference
    if reference_columns is None:
        reference_columns = columns
        print(f"Reference file: {file.name}")
        print(f"Number of columns: {len(columns)}")
        print()
#Flag any column differences
    if columns != reference_columns:
        all_match = False
        print(f"Column mismatch found in: {file.name}")
print("================================")
print("COLUMN CONSISTENCY CHECK")
print("================================")
if all_match:
    print("All CSV files have the same columns.")
else:
    print("Some CSV files have different columns.")
print(f"Files checked: {len(csv_files)}")