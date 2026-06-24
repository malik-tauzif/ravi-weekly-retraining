import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("weekly-readmission-model")

# Load THIS WEEK's data (the exact file we just versioned with DVC)
df = pd.read_csv("patient_data.csv")
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="weekly_retrain"):
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)

    # Link this run to the EXACT data version it used — closing the loop from Chapter 2
    with open("patient_data.csv.dvc") as f:
        dvc_content = f.read()
    data_hash = dvc_content.split("md5: ")[1].split("\n")[0]
    mlflow.set_tag("data_version_md5", data_hash)

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(model, artifact_path="model")

    # Also save a plain joblib copy, for our FastAPI server to load directly
    import joblib
    joblib.dump(model, "model.pkl")

    print(f"✅ Trained on data version {data_hash[:8]}... | Accuracy: {acc:.4f} | F1: {f1:.4f}")
