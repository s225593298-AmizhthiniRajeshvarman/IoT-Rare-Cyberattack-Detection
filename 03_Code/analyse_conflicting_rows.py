import pandas as pd
file_path = "../02_Dataset/working_dataset.csv"
#Load working dataset
df = pd.read_csv(file_path)
#Keep feature columns only
feature_columns = [
    col for col in df.columns
    if col != "Label"
]
print("Analysing conflicting feature combinations...")
#Count labels linked to each feature combination
label_counts = (
    df.groupby(feature_columns, dropna=False)["Label"]
    .transform("nunique")
)
#Keep rows that belong to conflicting combinations
conflicting_rows = df[label_counts > 1]
print("\n================================")
print("CONFLICTING ROW SUMMARY")
print("================================")
print(f"Total dataset rows: {len(df):,}")
print(f"Rows involved in label conflicts: {len(conflicting_rows):,}")
#Calculate affected percentage
percentage = (
    len(conflicting_rows) / len(df)
) * 100
print(f"Percentage of dataset affected: {percentage:.4f}%")
print(
    f"Unique labels involved: "
    f"{conflicting_rows['Label'].nunique()}"
)
print("\nMost affected labels:")
print(
    conflicting_rows["Label"]
    .value_counts()
    .head(15)
)
#Save conflicting rows for later checking
conflicting_rows.to_csv(
    "../04_Results/conflicting_rows.csv",
    index=False
)
print("\nSaved to:")
print("../04_Results/conflicting_rows.csv")