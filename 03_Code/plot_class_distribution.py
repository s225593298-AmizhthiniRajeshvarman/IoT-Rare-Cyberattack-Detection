import pandas as pd
import matplotlib.pyplot as plt
#Load class distribution results
df = pd.read_csv("../04_Results/class_distribution.csv")
#Sort classes by sample count
df = df.sort_values("Count", ascending=True)
#Create horizontal bar graph
plt.figure(figsize=(10, 12))
plt.barh(df["Label"], df["Count"])
plt.xlabel("Number of Samples")
plt.ylabel("Class")
plt.title("CICIoT2023 Class Distribution")
plt.tight_layout()
#Set output path
output_path = "../05_Figures/class_distribution.png"
#Save graph for report
plt.savefig(output_path, dpi=300)
print("Saved figure to:")
print(output_path)
plt.show()