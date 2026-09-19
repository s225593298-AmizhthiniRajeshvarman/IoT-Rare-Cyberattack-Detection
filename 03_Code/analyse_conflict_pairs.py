import pandas as pd
from collections import Counter
file_path = "../02_Dataset/working_dataset.csv"
#Load working dataset
df = pd.read_csv(file_path)
#Keep only feature columns
feature_columns = [
    col for col in df.columns
    if col != "Label"
]
print("Finding conflicting label combinations...")
#Group identical feature rows and collect labels
grouped = (
    df.groupby(feature_columns, dropna=False)["Label"]
    .agg(lambda x: tuple(sorted(set(x))))
)
#Keep groups linked to more than one label
conflicting_groups = grouped[
    grouped.apply(len) > 1
]
#Count how often each label combination appears
pair_counts = Counter(conflicting_groups)
print("\n================================")
print("MOST COMMON CONFLICTING LABEL SETS")
print("================================")
print(f"Total conflicting feature groups: {len(conflicting_groups):,}")
#Show 20 most common conflicts
for labels, count in pair_counts.most_common(20):
    print(f"{count:>6} groups : {'  <->  '.join(labels)}")