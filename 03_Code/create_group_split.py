import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
input_path = "../02_Dataset/working_dataset_deduplicated.csv"
#Load cleaned dataset
df = pd.read_csv(input_path)
#Keep feature columns only
feature_columns = [
    col for col in df.columns
    if col != "Label"
]
print("Creating feature-based groups...")
#Create one group for each identical feature combination
groups = pd.util.hash_pandas_object(
    df[feature_columns],
    index=False
)
#Create group-aware train and test split
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)
train_idx, test_idx = next(
    splitter.split(
        df,
        y=df["Label"],
        groups=groups
    )
)
#Create train and test datasets
train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()
print("\n================================")
print("GROUP-AWARE TRAIN / TEST SPLIT")
print("================================")
print(f"Total rows: {len(df):,}")
print(f"Training rows: {len(train_df):,}")
print(f"Testing rows: {len(test_df):,}")
print(
    f"Training percentage: "
    f"{len(train_df) / len(df) * 100:.2f}%"
)
print(
    f"Testing percentage: "
    f"{len(test_df) / len(df) * 100:.2f}%"
)
print(f"Training classes: {train_df['Label'].nunique()}")
print(f"Testing classes: {test_df['Label'].nunique()}")
#Check for feature-group overlap
train_groups = set(groups.iloc[train_idx])
test_groups = set(groups.iloc[test_idx])
overlap = train_groups.intersection(test_groups)
print(
    f"Feature-group overlap between train and test: "
    f"{len(overlap)}"
)
#Save split datasets
train_df.to_csv(
    "../02_Dataset/train_dataset.csv",
    index=False
)
test_df.to_csv(
    "../02_Dataset/test_dataset.csv",
    index=False
)
print("\nSaved:")
print("../02_Dataset/train_dataset.csv")
print("../02_Dataset/test_dataset.csv")