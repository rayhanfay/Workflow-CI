"""
modelling.py (versi MLflow Project untuk CI)
Dijalankan via:  mlflow run MLProject --env-manager=local
Tracking URI diatur lewat environment variable MLFLOW_TRACKING_URI (di workflow CI).
"""
import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

TARGET = "Outcome"

parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=300)
parser.add_argument("--max_depth", type=int, default=10)
parser.add_argument("--data_path", type=str, default="diabetes_preprocessing")
args = parser.parse_args()

data_dir = Path(args.data_path)
train = pd.read_csv(data_dir / "train.csv")
test = pd.read_csv(data_dir / "test.csv")
X_train, y_train = train.drop(columns=TARGET), train[TARGET]
X_test, y_test = test.drop(columns=TARGET), test[TARGET]

mlflow.sklearn.autolog(log_input_examples=True)

# `mlflow run` sudah membuat run aktif (MLFLOW_RUN_ID), start_run() akan melanjutkan run tersebut
with mlflow.start_run():
    model = RandomForestClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42
    )
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"Test accuracy: {acc:.4f}")
