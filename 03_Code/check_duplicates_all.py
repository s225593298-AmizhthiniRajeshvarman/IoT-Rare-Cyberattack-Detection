import pandas as pd
from pathlib import Path
#Set folder containing merged CSV files
dataset_folder = Path(
    r"C:\Users\ami25\Desktop\Resear\02_Dataset\MERGED_CSV\MERGED_CSV"
)
#Find all merged CSV files
csv_files = sorted(dataset_folder.glob("Merged*.csv"))
results = []
total_rows = 0
total_duplicates = 0
#Check duplicate rows in each file
for i, file in enumerate(csv_files, start=1):
    print(f"Checking {i}/{len(csv_files)}: {file.name}")
    df = pd.read_csv(file)
    rows = len(df)
    duplicates = df.duplicated().sum()
    percentage = (duplicates / rows) * 100
    total_rows += rows
    total_duplicates += duplicates
    #Store results for each file
    results.append({
        "File": file.name,
        "Rows": rows,
        "Duplicate_Rows": duplicates,
        "Duplicate_Percentage": percentage
    })
#Create summary table
results_df = pd.DataFrame(results)
results_df.to_csv(
    "../04_Results/duplicate_summary.csv",
    index=False
)
#Calculate overall duplicate percentage
overall_percentage = (total_duplicates / total_rows) * 100
print("\n================================")
print("FULL DATASET DUPLICATE SUMMARY")
print("================================")
print(f"Files checked: {len(csv_files)}")
print(f"Total rows: {total_rows:,}")
print(f"Duplicate rows within files: {total_duplicates:,}")
print(f"Duplicate percentage: {overall_percentage:.4f}%")
print("\nSaved to:")
print("../04_Results/duplicate_summary.csv")