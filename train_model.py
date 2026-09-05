import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    confusion_matrix
)

# -------------------------
# LOAD DATA
# -------------------------

print("\nLoading transaction data...")

df = pd.read_csv("data/transactions.csv")

# Features the AI will analyze
FEATURES = [
    "amount",
    "hour",
    "account_age_days",
    "transactions_last_hour",
    "failed_transactions",
    "distance_from_home"
]

X = df[FEATURES]
y = df["is_fraud"]

print(f"Total transactions: {len(df)}")
print(f"Fraud transactions: {y.sum()}")

# -------------------------
# SPLIT DATA
# -------------------------

# 70% training
# 15% validation
# 15% held-out testing

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nData split:")
print(f"Training: {len(X_train)}")
print(f"Validation: {len(X_val)}")
print(f"Held-out Test: {len(X_test)}")

# -------------------------
# TRAIN AI MODEL
# -------------------------

print("\nTraining FRISDI AI model...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# -------------------------
# VALIDATION
# -------------------------

print("\nTesting on held-out data...")

y_pred = model.predict(X_test)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

cm = confusion_matrix(y_test, y_pred)

print("\n==============================")
print("FRISDI MODEL PERFORMANCE")
print("==============================")

print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# -------------------------
# FALSE POSITIVE COST
# -------------------------

false_positives = cm[0][1]

COST_PER_FALSE_POSITIVE = 500

false_positive_cost = (
    false_positives * COST_PER_FALSE_POSITIVE
)

print("\n==============================")
print("FALSE POSITIVE COST")
print("==============================")

print(f"False positives: {false_positives}")
print(f"Estimated cost per false positive: ₹{COST_PER_FALSE_POSITIVE}")
print(f"Total estimated false-positive cost: ₹{false_positive_cost}")

# -------------------------
# SAVE MODEL
# -------------------------

model_package = {
    "model": model,
    "features": FEATURES,
    "precision": precision,
    "recall": recall,
    "false_positives": int(false_positives),
    "false_positive_cost": int(false_positive_cost),
    "test_size": int(len(X_test))
}

joblib.dump(
    model_package,
    "frisdi_model.pkl"
)

print("\nFRISDI model saved successfully!")
print("Model location: frisdi_model.pkl")
