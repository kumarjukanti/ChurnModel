"""Train churn prediction model"""

import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


# Load data
df = pd.read_csv("data/churn_data.csv")


# Features and target
features = [
    "age",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_support_calls",
]

X = df[features]
y = df["churn"]


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)

model.fit(X_train, y_train)


# Evaluate model
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"AUC-ROC: {auc:.4f}")


# Create models directory if it does not already exist
os.makedirs("models", exist_ok=True)


# Save trained model
model_path = "models/churn_model.pkl"

with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"Model saved to {model_path}")