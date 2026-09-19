import pandas as pd
input_path = "../02_Dataset/working_dataset.csv"
output_path = "../02_Dataset/working_dataset_deduplicated.csv"
print("Loading working dataset...")
#Load working dataset
df = pd.read_csv(input_path)
original_rows = len(df)
print("Removing exact duplicate rows...")
#Remove rows with identical values in all columns
df_clean = df.drop_duplicates().copy()
remaining_rows = len(df_clean)
removed_rows = original_rows - remaining_rows
print("\n================================")
print("EXACT DUPLICATE REMOVAL SUMMARY")
print("================================")
print(f"Original rows: {original_rows:,}")
print(f"Exact duplicates removed: {removed_rows:,}")
print(f"Rows remaining: {remaining_rows:,}")
print(
    f"Percentage removed: "
    f"{(removed_rows / original_rows) * 100:.4f}%"
)
print(f"Classes remaining: {df_clean['Label'].nunique()}")
#Save cleaned dataset
df_clean.to_csv(
    output_path,
    index=False
)
print("\nSaved to:")
print(output_path)