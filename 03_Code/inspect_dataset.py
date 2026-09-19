import pandas as pd
#Set file path
file_path = r"C:\Users\ami25\Desktop\Resear\02_Dataset\MERGED_CSV\MERGED_CSV\Merged01.csv"
#Load CSV file
df = pd.read_csv(file_path)
#Check dataset shape
print("Shape:")
print(df.shape)
#Show column names
print("\nColumns:")
print(df.columns.tolist())
#Show first few rows
print("\nFirst 5 rows:")
print(df.head())
#Check data types
print("\nData types:")
print(df.dtypes)
#Check missing values
print("\nMissing values:")
print(df.isnull().sum())
#Check label distribution
print("\nLabel counts:")
if "label" in df.columns:
    print(df["label"].value_counts())
elif "Label" in df.columns:
    print(df["Label"].value_counts())
else:
    print("No obvious label column found.")