
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

DATA_PATH = "archive(4).zip"
MODEL_PATH = "fraud_model.pkl"

def train_model():
    print("Loading transaction dataset...")
    data = pd.read_csv(DATA_PATH)

    if "Class" not in data.columns:
        raise ValueError("Dataset must contain a 'Class' column.")

    if data["Class"].isna().any():
        raise ValueError("Dataset contains missing target labels.")

    X = data.drop(columns=["Class"])
    y = data["Class"]

    if set(y.unique()) != {0, 1}:
        raise ValueError("Class must contain labels 0 (legitimate) and 1 (fraud).")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Legitimate", "Fraud"],
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")

    joblib.dump(
        {"model": model, "feature_names": list(X.columns)},
        MODEL_PATH
    )
    print(f"\nModel saved as {MODEL_PATH}")

def predict_transaction():
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    features = saved["feature_names"]

    print("\nEnter the values for one transaction.")
    print("Use the same feature order as the training CSV.")

    values = []
    for feature in features:
        while True:
            try:
                value = float(input(f"{feature}: "))
                values.append(value)
                break
            except ValueError:
                print("Please enter a valid number.")

    transaction = pd.DataFrame([values], columns=features)
    probability = model.predict_proba(transaction)[0, 1]

    print(f"\nEstimated fraud probability: {probability:.2%}")

    if model.predict(transaction)[0] == 1:
        print("⚠️ Flagged as potentially fraudulent.")
    else:
        print("Transaction classified as likely legitimate.")

if __name__ == "__main__":
    print("=== CREDIT CARD FRAUD DETECTION ===")
    print("1. Train model")
    print("2. Test a transaction")

    choice = input("Choose an option: ").strip()

    if choice == "1":
        train_model()
    elif choice == "2":
        predict_transaction()
    else:
        print("Invalid option.")