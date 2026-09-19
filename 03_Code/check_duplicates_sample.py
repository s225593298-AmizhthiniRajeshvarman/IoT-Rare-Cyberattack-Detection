import pandas as pd
#Set file path
file_path = r"C:\Users\ami25\Desktop\Resear\02_Dataset\MERGED_CSV\MERGED_CSV\Merged01.csv"
#Load CSV file
df = pd.read_csv(file_path)
#Count duplicate rows
duplicate_count = df.duplicated().sum()
print("================================")
print("DUPLICATE CHECK - Merged01.csv")
print("================================")
print(f"Total rows: {len(df):,}")
print(f"Duplicate rows: {duplicate_count:,}")
print(f"Duplicate percentage: {(duplicate_count / len(df)) * 100:.4f}%")