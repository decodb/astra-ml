import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier


DATA_PATH = "data/processed/traditional_metrics.csv"

AST_FEATURES = [
    "If",
    "For",
    "While",
    "Return",
    "TryCatch",
    "MethodCalls",
    "Variables",
    "Parameters",
    "ASTDepth",
    "BranchingFactor",
    "LOC"
]

TRADITIONAL_FEATURES = [
    "wmc",
    "dit",
    "noc",
    "cbo",
    "rfc",
    "lcom",
    "ca",
    "ce",
    "npm",
    "lcom3",
    "loc",
    "dam",
    "moa",
    "mfa",
    "cam",
    "ic",
    "cbm",
    "amc",
    "max_cc",
    "avg_cc"
]

FEATURES = AST_FEATURES + TRADITIONAL_FEATURES

TARGET = "Defective"
GROUP = "File"


def main():

    print("Loading combined dataset...")

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset: {len(df)} method rows")
    print(f"Features: {len(FEATURES)}")
    print(f"Files: {df[GROUP].nunique()}")

    print(
        f"Class distribution: "
        f"Clean {(df[TARGET] == 0).sum()}, "
        f"Defective {(df[TARGET] == 1).sum()}"
    )

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())

    y = df[TARGET].astype(int)
    groups = df[GROUP]

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    auc_scores = []
    f1_scores = []
    mcc_scores = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups), start=1
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        train_files = set(groups.iloc[train_idx])
        test_files = set(groups.iloc[test_idx])

        leakage = train_files.intersection(test_files)

        if leakage:
            print(f"WARNING: Leakage detected in fold {fold}")
        else:
            print(f"Fold {fold}: No file-level leakage")

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        auc = roc_auc_score(y_test, probabilities)
        f1 = f1_score(y_test, predictions)
        mcc = matthews_corrcoef(y_test, predictions)

        auc_scores.append(auc)
        f1_scores.append(f1)
        mcc_scores.append(mcc)

        print(
            f"Fold {fold}: "
            f"AUC={auc:.4f}, "
            f"F1={f1:.4f}, "
            f"MCC={mcc:.4f}"
        )

    print("\n=== AST + Traditional Metrics + Random Forest ===")

    print(
        f"ROC-AUC: "
        f"{np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}"
    )

    print(
        f"F1: "
        f"{np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}"
    )

    print(
        f"MCC: "
        f"{np.mean(mcc_scores):.4f} ± {np.std(mcc_scores):.4f}"
    )


if __name__ == "__main__":
    main()