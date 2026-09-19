import pandas as pd
import time
import pickle
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
#Load train and test data
print("Loading datasets...")
train_df = pd.read_csv("../02_Dataset/train_dataset.csv")
test_df = pd.read_csv("../02_Dataset/test_dataset.csv")
X_train = train_df.drop(columns=["Label"])
y_train = train_df["Label"]
X_test = test_df.drop(columns=["Label"])
y_test = test_df["Label"]
print(f"Training rows: {len(X_train):,}")
print(f"Testing rows: {len(X_test):,}")
print(f"Features: {X_train.shape[1]}")
#Create baseline Decision Tree
model = DecisionTreeClassifier(
    random_state=42
)
#Measure training time
print("\nTraining Decision Tree...")
start_train = time.perf_counter()
model.fit(X_train, y_train)
end_train = time.perf_counter()
training_time = end_train - start_train
#Measure prediction time
print("Running predictions...")
start_predict = time.perf_counter()
predictions = model.predict(X_test)
end_predict = time.perf_counter()
inference_time = end_predict - start_predict
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
    "../04_Results/baseline_decision_tree.pkl"
)
with open(model_path, "wb") as f:
    pickle.dump(model, f)
model_size_mb = (
    model_path.stat().st_size /
    (1024 * 1024)
)
#Save per-class results
report = classification_report(
    y_test,
    predictions,
    output_dict=True,
    zero_division=0
)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(
    "../04_Results/baseline_decision_tree_classification_report.csv"
)
#Save overall results
summary = pd.DataFrame([{
    "Model": "Decision Tree",
    "Imbalance_Method": "None",
    "Macro_Precision": macro_precision,
    "Macro_Recall": macro_recall,
    "Macro_F1": macro_f1,
    "Training_Time_Seconds": training_time,
    "Inference_Time_Seconds": inference_time,
    "Model_Size_MB": model_size_mb
}])
summary.to_csv(
    "../04_Results/baseline_decision_tree_summary.csv",
    index=False
)
#Display final results
print("\n================================")
print("BASELINE DECISION TREE RESULTS")
print("================================")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall:    {macro_recall:.4f}")
print(f"Macro F1:        {macro_f1:.4f}")
print(f"\nTraining time:   {training_time:.4f} seconds")
print(f"Inference time:  {inference_time:.4f} seconds")
print(f"Model size:      {model_size_mb:.4f} MB")
print("\nSaved:")
print("../04_Results/baseline_decision_tree_summary.csv")
print("../04_Results/baseline_decision_tree_classification_report.csv")
print("../04_Results/baseline_decision_tree.pkl")