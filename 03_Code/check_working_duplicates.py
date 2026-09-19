import pandas as pd
file_path = "../02_Dataset/working_dataset.csv"
#Load working dataset
df = pd.read_csv(file_path)
#Count duplicate rows
duplicate_count = df.duplicated().sum()
print("================================")
print("WORKING DATASET DUPLICATE CHECK")
print("================================")
print(f"Total rows: {len(df):,}")
print(f"Duplicate rows: {duplicate_count:,}")
print(
    f"Duplicate percentage: "
    f"{(duplicate_count / len(df)) * 100:.4f}%"
)
#Count unique rows
unique_rows = len(df.drop_duplicates())
print(f"Unique rows: {unique_rows:,}")