import pandas as pd
file_path = "../02_Dataset/working_dataset.csv"
#Load working dataset
df = pd.read_csv(file_path)
#Keep feature columns only
feature_columns = [
    col for col in df.columns
    if col != "Label"
]
print("Checking identical feature rows with different labels...")
#Count labels linked to each feature combination
label_counts = (
    df.groupby(feature_columns, dropna=False)["Label"]
    .nunique()
)
#Keep combinations linked to multiple labels
conflicting_groups = label_counts[label_counts > 1]
print("\n================================")
print("CONFLICTING DUPLICATE CHECK")
print("================================")
print(
    f"Feature combinations with multiple labels: "
    f"{len(conflicting_groups):,}"
)
if len(conflicting_groups) == 0:
    print("No conflicting duplicate feature rows found.")
else:
    print(
        "Some identical feature combinations have "
        "different class labels."
    )