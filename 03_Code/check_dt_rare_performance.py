import pandas as pd
report_path = "../04_Results/baseline_decision_tree_classification_report.csv"
#Load Decision Tree report
df = pd.read_csv(
    report_path,
    index_col=0
)
#Rare attacks selected for analysis
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
#Get results for rare classes
rare_results = df.loc[
    rare_classes,
    ["precision", "recall", "f1-score", "support"]
].copy()
print("================================")
print("DECISION TREE - RARE ATTACK PERFORMANCE")
print("================================")
print(rare_results.to_string())
print("\n================================")
print("RARE-CLASS AVERAGES")
print("================================")
#Calculate average rare-class scores
print(
    f"Average Precision: "
    f"{rare_results['precision'].mean():.4f}"
)
print(
    f"Average Recall:    "
    f"{rare_results['recall'].mean():.4f}"
)
print(
    f"Average F1:        "
    f"{rare_results['f1-score'].mean():.4f}"
)
#Save rare-class results
rare_results.to_csv(
    "../04_Results/baseline_decision_tree_rare_classes.csv"
)
print("\nSaved to:")
print("../04_Results/baseline_decision_tree_rare_classes.csv")