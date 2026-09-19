import pandas as pd
import matplotlib.pyplot as plt
#Load master comparison results
df = pd.read_csv(
    "../04_Results/master_model_comparison.csv"
)
#Fill missing baseline labels
df["Imbalance_Method"] = (
    df["Imbalance_Method"].fillna("None")
)
#Create label for each experiment
df["Experiment"] = (
    df["Model"] + " - " + df["Imbalance_Method"]
)
#Sort using rare-class F1 score
df = df.sort_values(
    "Rare_Avg_F1",
    ascending=True
)
#Create horizontal bar graph
plt.figure(figsize=(11, 7))
plt.barh(
    df["Experiment"],
    df["Rare_Avg_F1"]
)
plt.xlabel("Rare-Class Average F1 Score")
plt.ylabel("Model and Imbalance Method")
plt.title(
    "Rare Cyberattack Detection Performance"
)
plt.tight_layout()
#Save graph for report
plt.savefig(
    "../05_Figures/rare_f1_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
print(
    "Saved: ../05_Figures/rare_f1_comparison.png"
)