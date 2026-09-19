import pandas as pd
file_path = "../02_Dataset/working_dataset.csv"
#Load working dataset
df = pd.read_csv(file_path)
#Count samples for each class
counts = df["Label"].value_counts()
#Calculate class percentages
percentages = (
    counts / len(df) * 100
)
#Create class distribution table
results = pd.DataFrame({
    "Count": counts,
    "Percentage": percentages
})
print("================================")
print("WORKING DATASET CLASS DISTRIBUTION")
print("================================")
print(f"Total rows: {len(df):,}")
print(f"Total classes: {df['Label'].nunique()}")
print("\nClass distribution:")
print(results.to_string())
#Save class distribution results
results.to_csv(
    "../04_Results/working_class_distribution.csv"
)
print("\nSaved to:")
print("../04_Results/working_class_distribution.csv")