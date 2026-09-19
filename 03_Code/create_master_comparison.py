import pandas as pd
#Load comparison results for each model
dt = pd.read_csv(
    "../04_Results/decision_tree_comparison.csv"
)
lr = pd.read_csv(
    "../04_Results/logistic_regression_comparison.csv"
)
rf = pd.read_csv(
    "../04_Results/random_forest_comparison.csv"
)
lgbm = pd.read_csv(
    "../04_Results/lightgbm_comparison.csv"
)
#Combine all model results
master = pd.concat(
    [dt, lr, rf, lgbm],
    ignore_index=True
)
#Fill missing imbalance labels
master["Imbalance_Method"] = master["Imbalance_Method"].fillna("None")
#Save master comparison
master.to_csv(
    "../04_Results/master_model_comparison.csv",
    index=False
)
print("========================================")
print("MASTER MODEL COMPARISON")
print("========================================")
print(master.to_string(index=False))
print("\nSaved to:")
print("../04_Results/master_model_comparison.csv")