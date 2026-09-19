import pandas as pd
from pathlib import Path
#Set folder containing merged dataset files
dataset_folder = Path(
    r"C:\Users\ami25\Desktop\Resear\02_Dataset\MERGED_CSV\MERGED_CSV"
)
#Find all merged CSV files
csv_files = sorted(dataset_folder.glob("Merged*.csv"))
total_counts = {}
total_rows = 0
print(f"Files found: {len(csv_files)}")
#Read labels from each file and count classes
for i, file in enumerate(csv_files, start=1):
    print(f"Reading {i}/{len(csv_files)}: {file.name}")
    df = pd.read_csv(file, usecols=["Label"])
    total_rows += len(df)
    counts = df["Label"].value_counts()
    for label, count in counts.items():
        total_counts[label] = total_counts.get(label, 0) + count
#Put class counts into a dataframe
results = pd.DataFrame(
    total_counts.items(),
    columns=["Label", "Count"]
)
#Sort classes from most common to least common
results = results.sort_values(
    by="Count",
    ascending=False
).reset_index(drop=True)
#Calculate percentage for each class
results["Percentage"] = (
    results["Count"] / total_rows * 100
)
print("\n================================")
print("TOTAL DATASET SUMMARY")
print("================================")
print(f"\nTotal rows: {total_rows}")
print(f"Total classes: {len(results)}")
print("\nClass distribution:")
print(results.to_string(index=False))
#Save class distribution for analysis
results.to_csv(
    "../04_Results/class_distribution.csv",
    index=False
)
print("\nSaved to:")
print("../04_Results/class_distribution.csv")