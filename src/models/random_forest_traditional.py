import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    matthews_corrcoef
)


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "traditional_metrics.csv"
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("Loading traditional metrics dataset...")

df = pd.read_csv(DATA_FILE)

print("Dataset loaded.")
print("Total method rows:", len(df))


# --------------------------------------------------
# TRADITIONAL METRIC FEATURES
# --------------------------------------------------

FEATURES = [
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


TARGET = "Defective"

GROUP = "File"


# --------------------------------------------------
# CHECK REQUIRED COLUMNS
# --------------------------------------------------

required_columns = FEATURES + [
    TARGET,
    GROUP
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# --------------------------------------------------
# PREPARE FEATURES
# --------------------------------------------------

X = df[FEATURES].copy()

y = df[TARGET].astype(int)

groups = df[GROUP].astype(str)


# --------------------------------------------------
# HANDLE MISSING / NON-NUMERIC VALUES
# --------------------------------------------------

X = X.apply(
    pd.to_numeric,
    errors="coerce"
)

missing_values = X.isna().sum().sum()

if missing_values > 0:

    print(
        "Missing feature values found:",
        missing_values
    )

    # Replace missing values using
    # the median of each metric.
    X = X.fillna(
        X.median()
    )


# --------------------------------------------------
# DATASET SUMMARY
# --------------------------------------------------

print()
print("Features:", len(FEATURES))
print("Files:", groups.nunique())

print()
print("Class distribution:")

print(
    y.value_counts()
    .sort_index()
    .rename(
        index={
            0: "Clean",
            1: "Defective"
        }
    )
)


# --------------------------------------------------
# 5-FOLD FILE-LEVEL STRATIFIED CV
# --------------------------------------------------

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

auc_scores = []
f1_scores = []
mcc_scores = []


print()
print("=" * 60)
print("Traditional Metrics -> Random Forest")
print("5-Fold File-Level Cross-Validation")
print("=" * 60)


# --------------------------------------------------
# CROSS-VALIDATION
# --------------------------------------------------

for fold, (train_idx, test_idx) in enumerate(
    cv.split(
        X,
        y,
        groups=groups
    ),
    start=1
):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]


    # ----------------------------------------------
    # RANDOM FOREST
    # ----------------------------------------------

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )


    # ----------------------------------------------
    # TRAIN
    # ----------------------------------------------

    model.fit(
        X_train,
        y_train
    )


    # ----------------------------------------------
    # PREDICTIONS
    # ----------------------------------------------

    y_pred = model.predict(
        X_test
    )

    y_prob = model.predict_proba(
        X_test
    )[:, 1]


    # ----------------------------------------------
    # METRICS
    # ----------------------------------------------

    auc = roc_auc_score(
        y_test,
        y_prob
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_test,
        y_pred
    )


    # ----------------------------------------------
    # STORE RESULTS
    # ----------------------------------------------

    auc_scores.append(auc)
    f1_scores.append(f1)
    mcc_scores.append(mcc)


    # ----------------------------------------------
    # PRINT FOLD RESULTS
    # ----------------------------------------------

    train_files = groups.iloc[
        train_idx
    ].nunique()

    test_files = groups.iloc[
        test_idx
    ].nunique()


    print()
    print(f"Fold {fold}")
    print("-" * 30)

    print(
        f"Train methods: {len(train_idx)}"
    )

    print(
        f"Test methods:  {len(test_idx)}"
    )

    print(
        f"Train files:   {train_files}"
    )

    print(
        f"Test files:    {test_files}"
    )

    print(
        f"ROC-AUC:       {auc:.4f}"
    )

    print(
        f"F1:            {f1:.4f}"
    )

    print(
        f"MCC:           {mcc:.4f}"
    )


# --------------------------------------------------
# FINAL RESULTS
# --------------------------------------------------

print()
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(
    f"ROC-AUC: "
    f"{np.mean(auc_scores):.4f} "
    f"+/- "
    f"{np.std(auc_scores):.4f}"
)

print(
    f"F1:      "
    f"{np.mean(f1_scores):.4f} "
    f"+/- "
    f"{np.std(f1_scores):.4f}"
)

print(
    f"MCC:     "
    f"{np.mean(mcc_scores):.4f} "
    f"+/- "
    f"{np.std(mcc_scores):.4f}"
)


# --------------------------------------------------
# LEAKAGE CHECK
# --------------------------------------------------

print()
print("Checking for file-level leakage...")

leakage_found = False

for fold, (train_idx, test_idx) in enumerate(
    cv.split(
        X,
        y,
        groups=groups
    ),
    start=1
):

    train_files = set(
        groups.iloc[train_idx]
    )

    test_files = set(
        groups.iloc[test_idx]
    )

    overlap = (
        train_files &
        test_files
    )

    if overlap:

        leakage_found = True

        print(
            f"Fold {fold}: "
            f"{len(overlap)} overlapping files"
        )

    else:

        print(
            f"Fold {fold}: PASS"
        )


if leakage_found:

    print(
        "WARNING: File-level leakage detected."
    )

else:

    print(
        "PASS: No file-level leakage detected."
    )