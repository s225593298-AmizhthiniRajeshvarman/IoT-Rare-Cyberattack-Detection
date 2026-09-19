import pandas as pd
file_path = "../04_Results/class_distribution.csv"
#Load class distribution results
df = pd.read_csv(file_path)
#Set rare-class threshold
rare_threshold = 0.1
#Select rare attack classes
rare_classes = df[
    (df["Percentage"] < rare_threshold) &
    (df["Label"] != "BENIGN")
].copy()
print("================================")
print("RARE ATTACK CLASSES")
print("Threshold: < 0.1% of dataset")
print("================================\n")
print(rare_classes.to_string(index=False))
print(f"\nNumber of rare attack classes: {len(rare_classes)}")
print(
    f"Total rare attack samples: "
    f"{rare_classes['Count'].sum():,}"
)
print(
    f"Percentage of entire dataset: "
    f"{rare_classes['Percentage'].sum():.4f}%"
)
#Save rare attack list
rare_classes.to_csv(
    "../04_Results/rare_attack_classes.csv",
    index=False
)
print("\nSaved to:")
print("../04_Results/rare_attack_classes.csv")