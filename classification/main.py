import pandas as pd
import numpy as np
import json
import logging
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix
)

# This prints to the console and also saves the same output into logfile.txt
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler("logfile.txt"),
        logging.StreamHandler(sys.stdout)
    ]
)

def log_print(msg: str) -> None:
    """Helper function so I can log like print()."""
    logging.info(msg)

def restore_scaler_from_json(path: str) -> StandardScaler:
    """
    Restoring the StandardScaler exactly from scaler.json.
    """
    with open(path, "r") as f:
        d = json.load(f)

    sc = StandardScaler(with_mean=d["with_mean"], with_std=d["with_std"])

    # Restoring the learned parameters
    sc.mean_ = np.array(d["mean"], dtype=float)
    sc.scale_ = np.array(d["scale"], dtype=float)
    sc.var_ = np.array(d["var"], dtype=float)

    # Restoring sklearn metadata
    sc.n_features_in_ = int(d["n_features"])
    sc.feature_names_in_ = np.array(d["feature_names"])

    # prevents edge-case errors
    sc.n_samples_seen_ = np.array([1] * sc.n_features_in_, dtype=int)

    return sc

def ensure_feature_order(df: pd.DataFrame, expected_features: np.ndarray) -> pd.DataFrame:
    """
    Makes sure the dataframe has the exact same feature columns in the same order
    as the scaler expects.
    """
    expected = list(expected_features)

    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    extra = [c for c in df.columns if c not in expected]
    if extra:
        df = df.drop(columns=extra)

    return df[expected]

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute all metrics required by the assignment:
    Accuracy, Precision, Recall, F1, FDR, MCC
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    # FDR = FP / (TP + FP)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    denom = tp + fp
    fdr = (fp / denom) if denom > 0 else 0.0

    return {
        "Acc": acc,
        "Prec": prec,
        "Rec": rec,
        "F1": f1,
        "FDR": fdr,
        "MCC": mcc
    }

# Loading A.csv 
df = pd.read_csv("A.csv", sep=";")

# The target column is counterfeit
if "counterfeit" not in df.columns:
    raise ValueError("A.csv must contain a 'counterfeit' column.")

# y is the dependent variable
y = df["counterfeit"]

# X is all other columns
X = df.drop(columns=["counterfeit"])

# Spliting into 50% training and 50% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.50, random_state=42
)

log_print("Phase 2 Complete: Data loaded and split!")

# Restoring the StandardScaler from scaler.json (do NOT fit again)
scaler = restore_scaler_from_json("scaler.json")

# Making sure column order matches the scaler
X_train_ord = ensure_feature_order(X_train, scaler.feature_names_in_)
X_test_ord = ensure_feature_order(X_test, scaler.feature_names_in_)

# Scaling the features 
X_train_scaled = scaler.transform(X_train_ord)
X_test_scaled = scaler.transform(X_test_ord)

log_print("Phase 3 Complete: Scaler restored and data transformed!")

# Building the 8 required models using the assignment settings
model1 = LogisticRegression(tol=1, solver="liblinear", fit_intercept=False, max_iter=3)
model2 = KNeighborsClassifier(n_neighbors=3, p=2)
model3 = SVC(gamma="auto")
model4 = MLPClassifier(solver="lbfgs", alpha=0.1, hidden_layer_sizes=(1, 5))
model5 = DecisionTreeClassifier()
model6 = RandomForestClassifier(max_depth=2)
model7 = XGBClassifier(
    n_estimators=2, max_depth=2, learning_rate=1, objective="binary:logistic",
    eval_metric="logloss"
)
model8 = GaussianNB()

models = [model1, model2, model3, model4, model5, model6, model7, model8]

model_names = ["model1", "model2", "model3", "model4", "model5", "model6", "model7", "model8"]

# Training all 8 models using the scaled training data
for m in models:
    m.fit(X_train_scaled, y_train)

log_print("Phase 4 Complete: All 8 models trained!")

# Predicting on D.csv
df_d = pd.read_csv("D.csv", sep=";")

# Making sure D.csv feature order matches the scaler
df_d_ord = ensure_feature_order(df_d, scaler.feature_names_in_)
X_d_scaled = scaler.transform(df_d_ord)

log_print("\n--- Predictions for D.csv ---")

all_preds = []
for name, m in zip(model_names, models):
    pred = m.predict(X_d_scaled).astype(int)
    all_preds.append(pred)

    # Print predictions for both D.csv samples
    log_print(f"{name} Predicted Labels: {pred.tolist()}")

# Consensus prediction using majority vote
all_preds_np = np.vstack(all_preds)  # shape: (8, 2)

consensus = []
for sample_idx in range(all_preds_np.shape[1]):
    votes = all_preds_np[:, sample_idx]
    majority = int(np.argmax(np.bincount(votes)))
    consensus.append(majority)

log_print(f"Consensus Prediction for D.csv: {consensus}")

# Evaluate models using E.csv 
df_e = pd.read_csv("E.csv", sep=";")

if "counterfeit" not in df_e.columns:
    raise ValueError("E.csv must contain a 'counterfeit' column.")

y_e = df_e["counterfeit"].to_numpy()
X_e = df_e.drop(columns=["counterfeit"])

# Making sure E.csv feature order matches the scaler
X_e_ord = ensure_feature_order(X_e, scaler.feature_names_in_)

# Scale E.csv features
X_e_scaled = scaler.transform(X_e_ord)

log_print("\n--- Evaluation Metrics for E.csv ---")

for name, m in zip(model_names, models):
    y_pred = m.predict(X_e_scaled).astype(int)
    met = compute_metrics(y_e, y_pred)

    log_print(
        f"{name}: "
        f"Acc={met['Acc']:.4f}, Prec={met['Prec']:.4f}, Rec={met['Rec']:.4f}, "
        f"F1={met['F1']:.4f}, FDR={met['FDR']:.4f}, MCC={met['MCC']:.4f}"
    )




