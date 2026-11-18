"""Reproduce the documented train/test comparison on the checked-in synthetic sample."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "synthetic_microloan_sample.csv"
OUTPUT_PATH = ROOT / "reports" / "reproduced_model_metrics.csv"
THRESHOLD_PATH = ROOT / "docs" / "data" / "threshold_scores.json"
SEED = 4_250_779_039
K = 3
TREE_DEPTH = 4
TARGET = "default"


def fit_preprocessor(X_fit, X_apply):
    """Fit median imputation and scaling on fit data only; apply without leakage."""
    imputer = SimpleImputer(strategy="median")
    X_fit_imp = imputer.fit_transform(X_fit)
    X_apply_imp = imputer.transform(X_apply)
    scaler = StandardScaler()
    X_fit_scaled = scaler.fit_transform(X_fit_imp)
    X_apply_scaled = scaler.transform(X_apply_imp)
    return X_fit_imp, X_apply_imp, X_fit_scaled, X_apply_scaled


def main():
    df = pd.read_csv(DATA_PATH, index_col="master_row_id")
    X, y = df.drop(columns=TARGET), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    Xtr_imp, Xte_imp, Xtr_sc, Xte_sc = fit_preprocessor(X_train, X_test)

    models = {
        "Baseline (majority)": (DummyClassifier(strategy="most_frequent").fit(Xtr_sc, y_train), Xte_sc),
        "Logistic regression": (LogisticRegression(max_iter=1000).fit(Xtr_sc, y_train), Xte_sc),
        "KNN": (KNeighborsClassifier(n_neighbors=K).fit(Xtr_sc, y_train), Xte_sc),
        "Naive Bayes": (GaussianNB().fit(Xtr_imp, y_train), Xte_imp),
        "Decision tree": (DecisionTreeClassifier(criterion="entropy", max_depth=TREE_DEPTH,
                                                  random_state=SEED).fit(Xtr_imp, y_train), Xte_imp),
        "Linear SVM": (SVC(kernel="linear", C=1.0).fit(Xtr_sc, y_train), Xte_sc),
    }
    # Save the held-out labels and Logistic Regression probabilities so the static\n    # portfolio page can calculate exact metrics at any interactive threshold.\n    lr_model, lr_X_eval = models["Logistic regression"]\n    threshold_payload = {\n        "source": "held-out test split",\n        "y_true": y_test.astype(int).tolist(),\n        "probabilities": lr_model.predict_proba(lr_X_eval)[:, 1].tolist(),\n    }\n    THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)\n    THRESHOLD_PATH.write_text(json.dumps(threshold_payload), encoding="utf-8")\n\n    rows = []\n    for name, (model, X_eval) in models.items():
        pred = model.predict(X_eval)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        scores = model.decision_function(X_eval) if name == "Linear SVM" else model.predict_proba(X_eval)[:, 1]
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, scores),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        })
    results = pd.DataFrame(rows).set_index("model")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, float_format="%.6f")
    print(results.round(3).to_string())
    print(f"\nSaved: {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
