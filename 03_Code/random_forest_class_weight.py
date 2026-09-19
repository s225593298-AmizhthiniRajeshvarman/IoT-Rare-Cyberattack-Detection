import pandas as pd
import time
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
print("Loading datasets...")
#Load train and test data
train_df = pd.read_csv("../02_Dataset/train_dataset.csv")
test_df = pd.read_csv("../02_Dataset/test_dataset.csv")
#Separate features and labels
X_train = train_df.drop(columns=["Label"])
y_train = train_df["Label"]
X_test = test_df.drop(columns=["Label"])
y_test = test_df["Label"]
print(f"Training rows: {len(X_train):,}")
print(f"Testing rows: {len(X_test):,}")
print(f"Features: {X_train.shape[1]}")
#Create class-weighted Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
print("\nTraining class-weighted Random Forest...")
#Measure training time
start_train = time.perf_counter()
model.fit(X_train, y_train)
training_time = time.perf_counter() - start_train
print("Running predictions...")
#Measure prediction time
start_predict = time.perf_counter()
predictions = model.predict(X_test)
inference_time = time.perf_counter() - start_predict
#Calculate macro scores
macro_precision = precision_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)
macro_recall = recall_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)
macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)
#Save trained model
model_path = Path(
    "../04_Results/random_forest_class_weight.pkl"
)
with open(model_path, "wb") as f:
    pickle.dump(model, f)
model_size_mb = (
    model_path.stat().st_size /
    (1024 * 1024)
)
#Get results for each class
report = classification_report(
    y_test,
    predictions,
    output_dict=True,
    zero_division=0
)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(
    "../04_Results/random_forest_class_weight_report.csv"
)
#Rare attacks used for separate analysis
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
rare_results = report_df.loc[
    rare_classes,
    [
        "precision",
        "recall",
        "f1-score",
        "support"
    ]
]
#Calculate average rare-class results
rare_precision = rare_results["precision"].mean()
rare_recall = rare_results["recall"].mean()
rare_f1 = rare_results["f1-score"].mean()
rare_results.to_csv(
    "../04_Results/random_forest_class_weight_rare_classes.csv"
)
#Store main results for comparison
summary = pd.DataFrame([{
    "Model": "Random Forest",
    "Imbalance_Method": "Class Weighting",
    "Macro_Precision": macro_precision,
    "Macro_Recall": macro_recall,
    "Macro_F1": macro_f1,
    "Rare_Avg_Precision": rare_precision,
    "Rare_Avg_Recall": rare_recall,
    "Rare_Avg_F1": rare_f1,
    "Training_Time_Seconds": training_time,
    "Inference_Time_Seconds": inference_time,
    "Model_Size_MB": model_size_mb
}])
summary.to_csv(
    "../04_Results/random_forest_class_weight_summary.csv",
    index=False
)
#Display final results
print("\n================================")
print("RANDOM FOREST + CLASS WEIGHTING")
print("================================")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall:    {macro_recall:.4f}")
print(f"Macro F1:        {macro_f1:.4f}")
print("\nRare-class averages:")
print(f"Rare Precision:  {rare_precision:.4f}")
print(f"Rare Recall:     {rare_recall:.4f}")
print(f"Rare F1:         {rare_f1:.4f}")
print(f"\nTraining time:   {training_time:.4f} seconds")
print(f"Inference time:  {inference_time:.4f} seconds")
print(f"Model size:      {model_size_mb:.4f} MB")
print("\nRare attack results:")
print(rare_results.to_string())