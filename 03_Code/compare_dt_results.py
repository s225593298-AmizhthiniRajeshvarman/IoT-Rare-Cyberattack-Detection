import pandas as pd
#Load baseline results
baseline_summary = pd.read_csv(
    "../04_Results/baseline_decision_tree_summary.csv"
)
baseline_rare = pd.read_csv(
    "../04_Results/baseline_decision_tree_rare_classes.csv",
    index_col=0
)
#Load imbalance method results
weighted_summary = pd.read_csv(
    "../04_Results/decision_tree_class_weight_summary.csv"
)
undersampling_summary = pd.read_csv(
    "../04_Results/decision_tree_undersampling_summary.csv"
)
smote_summary = pd.read_csv(
    "../04_Results/decision_tree_smote_summary.csv"
)
#Combine Decision Tree results
comparison = pd.DataFrame([
    {
        "Model": "Decision Tree",
        "Imbalance_Method": "None",
        "Macro_F1": baseline_summary.loc[0, "Macro_F1"],
        "Rare_Avg_F1": baseline_rare["f1-score"].mean(),
        "Training_Time_Seconds":
            baseline_summary.loc[0, "Training_Time_Seconds"],
        "Inference_Time_Seconds":
            baseline_summary.loc[0, "Inference_Time_Seconds"],
        "Model_Size_MB":
            baseline_summary.loc[0, "Model_Size_MB"]
    },
    {
        "Model": "Decision Tree",
        "Imbalance_Method": "Class Weighting",
        "Macro_F1": weighted_summary.loc[0, "Macro_F1"],
        "Rare_Avg_F1": weighted_summary.loc[0, "Rare_Avg_F1"],
        "Training_Time_Seconds":
            weighted_summary.loc[0, "Training_Time_Seconds"],
        "Inference_Time_Seconds":
            weighted_summary.loc[0, "Inference_Time_Seconds"],
        "Model_Size_MB":
            weighted_summary.loc[0, "Model_Size_MB"]
    },
    {
        "Model": "Decision Tree",
        "Imbalance_Method": "Random Undersampling",
        "Macro_F1": undersampling_summary.loc[0, "Macro_F1"],
        "Rare_Avg_F1": undersampling_summary.loc[0, "Rare_Avg_F1"],
        "Training_Time_Seconds":
            undersampling_summary.loc[0, "Training_Time_Seconds"],
        "Inference_Time_Seconds":
            undersampling_summary.loc[0, "Inference_Time_Seconds"],
        "Model_Size_MB":
            undersampling_summary.loc[0, "Model_Size_MB"]
    },
    {
        "Model": "Decision Tree",
        "Imbalance_Method": "SMOTE",
        "Macro_F1": smote_summary.loc[0, "Macro_F1"],
        "Rare_Avg_F1": smote_summary.loc[0, "Rare_Avg_F1"],
        "Training_Time_Seconds":
            smote_summary.loc[0, "Training_Time_Seconds"],
        "Inference_Time_Seconds":
            smote_summary.loc[0, "Inference_Time_Seconds"],
        "Model_Size_MB":
            smote_summary.loc[0, "Model_Size_MB"]
    }
])
#Save comparison results
comparison.to_csv(
    "../04_Results/decision_tree_comparison.csv",
    index=False
)
print("================================")
print("FINAL DECISION TREE COMPARISON")
print("================================")
print(comparison.to_string(index=False))
print("\nSaved to:")
print("../04_Results/decision_tree_comparison.csv")