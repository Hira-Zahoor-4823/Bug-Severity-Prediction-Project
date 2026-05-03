# DS4SE Project: Milestone 3: Data Modelling and Evaluation 
# - Predicting bug severity and priority, exporting metrics to CSV, and generating Confusion Matrices

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("Step 1: Loading Dataset")
print("=" * 60)
df = pd.read_csv('frontend_uiux_bug_dataset_cleaned.csv')
print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.\n")

print("=" * 60)
print("Step 2: Data Preprocessing & Encoding")
print("=" * 60)
cols_to_drop = ['bug_id', 'steps_to_reproduce', 'expected_behavior', 'actual_behavior', 'tags']
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

categorical_cols = ['app_name', 'module', 'bug_type', 'device_type', 'browser', 'os', 'reported_by', 'screenshot_available', 'resolved']
encoders_dict = {}
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders_dict[col] = le 

X = df.drop(columns=['severity', 'priority'])
y_severity = df['severity']
y_priority = df['priority']

le_sev = LabelEncoder()
le_pri = LabelEncoder()
y_severity_encoded = le_sev.fit_transform(y_severity)
y_priority_encoded = le_pri.fit_transform(y_priority)
print("Categorical features and targets successfully converted to numerical formats.\n")

print("=" * 60)
print("Step 3: Train-Test Split (80% Training, 20% Testing)")
print("=" * 60)
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y_severity_encoded, test_size=0.2, random_state=42)
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X, y_priority_encoded, test_size=0.2, random_state=42)

# Separate model dictionaries to prevent overwriting during training
models_severity = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=50),
    "Naive Bayes": GaussianNB()
}

models_priority = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=50),
    "Naive Bayes": GaussianNB()
}

metrics_data = []

print("=" * 60)
print("Step 4: Training Models for SEVERITY")
print("=" * 60)
for name, model in models_severity.items():
    model.fit(X_train_s, y_train_s)
    preds = model.predict(X_test_s)
    
    acc = accuracy_score(y_test_s, preds)
    report_dict = classification_report(y_test_s, preds, target_names=le_sev.classes_, output_dict=True)
    
    print(f"➔ {name} Accuracy: {acc * 100:.2f}%")
    
    for class_name, metrics in report_dict.items():
        if isinstance(metrics, dict):
            metrics_data.append({
                'Target Variable': 'Severity', 'Algorithm': name, 'Metric Type': class_name,
                'Precision': round(metrics['precision'], 4), 'Recall': round(metrics['recall'], 4),
                'F1-Score': round(metrics['f1-score'], 4), 'Support (Test Samples)': metrics['support']
            })
            
    # Generate Confusion Matrix  for Random Forest Severity
    if name == "Random Forest":
        cm_sev = confusion_matrix(y_test_s, preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_sev, annot=True, fmt='d', cmap='Blues', xticklabels=le_sev.classes_, yticklabels=le_sev.classes_)
        plt.title('Confusion Matrix - Severity Prediction (Random Forest)')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix_severity.png', dpi=150)
        plt.close()

print("\n" + "=" * 60)
print("Step 5: Training Models for PRIORITY")
print("=" * 60)
for name, model in models_priority.items():
    model.fit(X_train_p, y_train_p)
    preds = model.predict(X_test_p)
    
    acc = accuracy_score(y_test_p, preds)
    report_dict = classification_report(y_test_p, preds, target_names=le_pri.classes_, output_dict=True)
    
    print(f"➔ {name} Accuracy: {acc * 100:.2f}%")
    
    for class_name, metrics in report_dict.items():
        if isinstance(metrics, dict):
            metrics_data.append({
                'Target Variable': 'Priority', 'Algorithm': name, 'Metric Type': class_name,
                'Precision': round(metrics['precision'], 4), 'Recall': round(metrics['recall'], 4),
                'F1-Score': round(metrics['f1-score'], 4), 'Support (Test Samples)': metrics['support']
            })
            
    #Confusion Matrix for Random Forest Priority
    if name == "Random Forest":
        cm_pri = confusion_matrix(y_test_p, preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_pri, annot=True, fmt='d', cmap='Oranges', xticklabels=le_pri.classes_, yticklabels=le_pri.classes_)
        plt.title('Confusion Matrix - Priority Prediction (Random Forest)')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix_priority.png', dpi=150)
        plt.close()

print("\n" + "=" * 60)
print("Step 6: Exporting Metrics to CSV")
print("=" * 60)
metrics_df = pd.DataFrame(metrics_data)
output_filename = "model_evaluation_metrics.csv"
metrics_df.to_csv(output_filename, index=False)

print(f"SUCCESS! All metrics and tables have been saved to: '{output_filename}'")
print("SUCCESS! Confusion matrices saved as 'confusion_matrix_severity.png' and 'confusion_matrix_priority.png'")

print("\n" + "=" * 60)
print("Step 7: Exporting Models for Interactive Predictions")
print("=" * 60)

joblib.dump(models_severity["Random Forest"], "severity_rf_model.joblib")
joblib.dump(models_priority["Random Forest"], "priority_rf_model.joblib")

export_data = {
    'features': encoders_dict,
    'target_severity': le_sev,
    'target_priority': le_pri
}
joblib.dump(export_data, "data_encoders.joblib")

print("SUCCESS! 'severity_rf_model.joblib', 'priority_rf_model.joblib', and 'data_encoders.joblib' generated.")
print("Milestone 3 Completed Successfully!")
