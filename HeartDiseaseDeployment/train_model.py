"""
train_model.py
----------------
Heart Disease Prediction - Data Understanding, Preprocessing & Model Training
(Task 1 & Task 2 of the assignment)

Dataset columns:
age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak,
slope, ca, thal, target

target: 1 = heart disease present, 0 = no heart disease
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------------------------
# TASK 1: Data Understanding and Preprocessing
# ---------------------------------------------------------------------------

# 1. Load the dataset using Pandas
df = pd.read_csv("heart.csv")

# 2. Display the first five records
print("First 5 records:")
print(df.head(), "\n")

# 3. Identify numerical features and the target variable
TARGET = "target"
numerical_features = [col for col in df.columns if col != TARGET]
print("Numerical features:", numerical_features)
print("Target variable:", TARGET, "\n")

# 4. Check for missing values
print("Missing values per column:")
print(df.isnull().sum(), "\n")

# 5. Split the dataset into 80% training and 20% testing
X = df[numerical_features]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}\n")

# ---------------------------------------------------------------------------
# TASK 2: Model Development
# ---------------------------------------------------------------------------

# Feature scaling (helps consistency of input, especially for future model swaps)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Classifier
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------------------------------------------------------------------------
# Save the trained model and scaler using Joblib
# ---------------------------------------------------------------------------
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(numerical_features, "feature_names.pkl")

print("\nSaved model.pkl, scaler.pkl and feature_names.pkl successfully.")
