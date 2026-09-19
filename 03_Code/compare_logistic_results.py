import pandas as pd
#Load baseline results
baseline = pd.read_csv(
    "../04_Results/baseline_logistic_regression_summary.csv"
)
#Load imbalance method results
weighted = pd.read_csv(
    "../04_Results/logistic_regression_class_weight_summary.csv"
)
smote = pd.read_csv(
    "../04_Results/logistic_regression_smote_summary.csv"
)
#Combine Logistic Regression results
comparison = pd.DataFrame([
    {
        "Model": "Logistic Regression",
        "Imbalance_Method": "None",
        "Macro_F1": baseline.loc[0, "Macro_F1"],
        "Rare_Avg_F1": baseline.loc[0, "Rare_Avg_F1"],
        "Training_Time_Seconds":
            baseline.loc[0, "Training_Time_Seconds"],
        "Inference_Time_Seconds":
            baseline.loc[0, "Inference_Time_Seconds"],
        "Model_Size_MB":
            baseline.loc[0, "Model_Size_MB"]
    },
    {
        "Model": "Logistic Regression",
        "Imbalance_Method": "Class Weighting",
        "Macro_F1": weighted.loc[0, "Macro_F1"],
        "Rare_Avg_F1": weighted.loc[0, "Rare_Avg_F1"],
        "Training_Time_Seconds":
            weighted.loc[0, "Training_Time_Seconds"],
        "Inference_Time_Seconds":
            weighted.loc[0, "Inference_Time_Seconds"],
        "Model_Size_MB":
            weighted.loc[0, "Model_Size_MB"]
    },
    {
        "Model": "Logistic Regression",
        "Imbalance_Method": "SMOTE",
        "Macro_F1": smote.loc[0, "Macro_F1"],
        "Rare_Avg_F1": smote.loc[0, "Rare_Avg_F1"],
        "Training_Time_Seconds":
            smote.loc[0, "Training_Time_Seconds"],
        "Inference_Time_Seconds":
            smote.loc[0, "Inference_Time_Seconds"],
        "Model_Size_MB":
            smote.loc[0, "Model_Size_MB"]
    }
])
#Save comparison results
comparison.to_csv(
    "../04_Results/logistic_regression_comparison.csv",
    index=False
)
print("================================")
print("FINAL LOGISTIC REGRESSION COMPARISON")
print("================================")
print(comparison.to_string(index=False))
print("\nSaved to:")
print("../04_Results/logistic_regression_comparison.csv")