import pandas as pd
import time
import pickle
from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from imblearn.over_sampling import SMOTE
print("Loading datasets...")
#Load train and test data
train_df = pd.read_csv("../02_Dataset/train_dataset.csv")
test_df = pd.read_csv("../02_Dataset/test_dataset.csv")
#Separate features and labels
X_train = train_df.drop(columns=["Label"])
y_train_text = train_df["Label"]
X_test = test_df.drop(columns=["Label"])
y_test_text = test_df["Label"]
print(f"Original training rows: {len(X_train):,}")
print(f"Testing rows: {len(X_test):,}")
print(f"Features: {X_train.shape[1]}")
#Convert labels into numbers
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_text)
y_test = label_encoder.transform(y_test_text)
print(f"Classes: {len(label_encoder.classes_)}")
#Set up SMOTE
smote = SMOTE(
    random_state=42,
    k_neighbors=5
)
print("\nApplying SMOTE...")
#Measure SMOTE processing time
start_smote = time.perf_counter()
X_resampled, y_resampled = smote.fit_resample(
    X_train,
    y_train
)
smote_time = time.perf_counter() - start_smote
print(f"SMOTE time: {smote_time:.4f} seconds")
print(f"Resampled training rows: {len(X_resampled):,}")
print(f"Classes after SMOTE: {len(set(y_resampled))}")
#Create LightGBM model
model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)
print("\nTraining LightGBM with SMOTE...")
#Measure training time
start_train = time.perf_counter()
model.fit(
    X_resampled,
    y_resampled
)
training_time = time.perf_counter() - start_train
print("Running predictions...")
#Measure prediction time
start_predict = time.perf_counter()
predictions_encoded = model.predict(X_test)
inference_time = time.perf_counter() - start_predict
#Convert predictions back to attack names
predictions = label_encoder.inverse_transform(
    predictions_encoded.astype(int)
)
#Calculate macro scores
macro_precision = precision_score(
    y_test_text,
    predictions,
    average="macro",
    zero_division=0
)
macro_recall = recall_score(
    y_test_text,
    predictions,
    average="macro",
    zero_division=0
)
macro_f1 = f1_score(
    y_test_text,
    predictions,
    average="macro",
    zero_division=0
)
#Save trained model
model_path = Path(
    "../04_Results/lightgbm_smote.pkl"
)
with open(model_path, "wb") as f:
    pickle.dump(
        {
            "model": model,
            "label_encoder": label_encoder
        },
        f
    )
model_size_mb = (
    model_path.stat().st_size /
    (1024 * 1024)
)
#Get results for each class
report = classification_report(
    y_test_text,
    predictions,
    output_dict=True,
    zero_division=0
)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(
    "../04_Results/lightgbm_smote_report.csv"
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
    ["precision", "recall", "f1-score", "support"]
]
#Calculate average rare-class results
rare_precision = rare_results["precision"].mean()
rare_recall = rare_results["recall"].mean()
rare_f1 = rare_results["f1-score"].mean()
rare_results.to_csv(
    "../04_Results/lightgbm_smote_rare_classes.csv"
)
#Store main results for comparison
summary = pd.DataFrame([{
    "Model": "LightGBM",
    "Imbalance_Method": "SMOTE",
    "Original_Training_Rows": len(X_train),
    "Resampled_Training_Rows": len(X_resampled),
    "SMOTE_Time_Seconds": smote_time,
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
    "../04_Results/lightgbm_smote_summary.csv",
    index=False
)
#Display final results
print("\n================================")
print("LIGHTGBM + SMOTE")
print("================================")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall:    {macro_recall:.4f}")
print(f"Macro F1:        {macro_f1:.4f}")
print("\nRare-class averages:")
print(f"Rare Precision:  {rare_precision:.4f}")
print(f"Rare Recall:     {rare_recall:.4f}")
print(f"Rare F1:         {rare_f1:.4f}")
print(f"\nSMOTE time:      {smote_time:.4f} seconds")
print(f"Training time:   {training_time:.4f} seconds")
print(f"Inference time:  {inference_time:.4f} seconds")
print(f"Model size:      {model_size_mb:.4f} MB")
print("\nRare attack results:")
print(rare_results.to_string())