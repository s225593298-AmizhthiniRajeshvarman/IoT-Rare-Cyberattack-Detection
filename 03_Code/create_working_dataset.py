import pandas as pd
import numpy as np
from pathlib import Path
#Set folder containing merged CSV files
dataset_folder = Path(
    r"C:\Users\ami25\Desktop\Resear\02_Dataset\MERGED_CSV\MERGED_CSV"
)
#Find all merged CSV files
csv_files = sorted(dataset_folder.glob("Merged*.csv"))
TARGET_TOTAL = 1_000_000
#Load labels first to calculate class proportions
label_frames = []
print("Reading labels from all files...")
for i, file in enumerate(csv_files, start=1):
    print(f"Labels {i}/{len(csv_files)}: {file.name}")
    labels = pd.read_csv(file, usecols=["Label"])
    label_frames.append(labels)
all_labels = pd.concat(label_frames, ignore_index=True)
#Count samples for each class
class_counts = all_labels["Label"].value_counts()
print("\nClass counts calculated.")
sample_targets = {}
#Calculate target size for each class
for label, count in class_counts.items():
    proportion = count / len(all_labels)
    target_count = int(proportion * TARGET_TOTAL)
    #Keep very rare classes completely
    if count < 50000:
        target_count = count
    sample_targets[label] = min(target_count, count)
print("\nTarget sample sizes calculated.")
sampled_parts = []
#Sample data from each file
for i, file in enumerate(csv_files, start=1):
    print(f"Sampling {i}/{len(csv_files)}: {file.name}")
    df = pd.read_csv(file)
    #Remove invalid values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    for label in df["Label"].unique():
        class_df = df[df["Label"] == label]
        total_class_count = class_counts[label]
        desired_total = sample_targets[label]
        fraction = desired_total / total_class_count
        sample_size = int(len(class_df) * fraction)
        if sample_size > 0:
            sample = class_df.sample(
                n=min(sample_size, len(class_df)),
                random_state=42
            )
            sampled_parts.append(sample)
#Combine sampled data
working_df = pd.concat(sampled_parts, ignore_index=True)
#Shuffle working dataset
working_df = working_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)
output_path = "../02_Dataset/working_dataset.csv"
#Save working dataset
working_df.to_csv(output_path, index=False)
print("\n================================")
print("WORKING DATASET CREATED")
print("================================")
print(f"Rows: {len(working_df):,}")
print(f"Columns: {working_df.shape[1]}")
print(f"Classes: {working_df['Label'].nunique()}")
print("\nRare class counts:")
rare_classes = [
    "DDOS-HTTP_FLOOD",
    "DDOS-SLOWLORIS",
    "DICTIONARYBRUTEFORCE",
    "BROWSERHIJACKING",
    "COMMANDINJECTION",
    "SQLINJECTION",
    "XSS",
    "BACKDOOR_MALWARE",
    "RECON-PINGSWEEP",
    "UPLOADING_ATTACK"
]
#Show rare attack counts
print(
    working_df[
        working_df["Label"].isin(rare_classes)
    ]["Label"].value_counts()
)
print("\nSaved to:")
print(output_path)