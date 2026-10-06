import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    matthews_corrcoef
)


INPUT_FILE = "data/raw/ast_features_labeled.csv"


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded successfully.")
print(f"Number of methods: {len(df)}")
print(f"Number of files: {df['File'].nunique()}")


# --------------------------------------------------
# AST FEATURES
# --------------------------------------------------

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


X = df[AST_FEATURES]
y = df["Defective"]

groups = df["File"]


# --------------------------------------------------
# DATASET INFORMATION
# --------------------------------------------------

print("\n----------------------------------------")
print("DATASET INFORMATION")
print("----------------------------------------")

print(f"Methods: {len(df)}")
print(f"Files:   {df['File'].nunique()}")

print("\nOverall class distribution:")
print(y.value_counts())

print("\nOverall class percentages:")
print(y.value_counts(normalize=True) * 100)


# --------------------------------------------------
# CROSS-VALIDATION
# --------------------------------------------------

N_SPLITS = 5

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=42
)


auc_scores = []
f1_scores = []
mcc_scores = []


print("\n========================================")
print("XGBOOST FILE-LEVEL CROSS-VALIDATION")
print("========================================")


for fold, (train_indices, test_indices) in enumerate(
        cv.split(X, y, groups=groups), start=1):

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]

    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]


    # --------------------------------------------------
    # LEAKAGE CHECK
    # --------------------------------------------------

    train_files = set(
        df.iloc[train_indices]["File"]
    )

    test_files = set(
        df.iloc[test_indices]["File"]
    )

    overlap = train_files.intersection(test_files)


    print("\n----------------------------------------")
    print(f"FOLD {fold}")
    print("----------------------------------------")

    print(f"Training methods: {len(X_train)}")
    print(f"Testing methods:  {len(X_test)}")

    print(f"Training files: {len(train_files)}")
    print(f"Testing files:  {len(test_files)}")


    print("\nTraining class distribution:")
    print(y_train.value_counts().to_dict())

    print("\nTesting class distribution:")
    print(y_test.value_counts().to_dict())


    if len(overlap) == 0:
        print("\nLeakage check: PASS")
    else:
        print("\nLeakage check: FAIL")
        print("Overlapping files:", overlap)


    # --------------------------------------------------
    # XGBOOST MODEL
    # --------------------------------------------------

    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )


    # --------------------------------------------------
    # TRAIN
    # --------------------------------------------------

    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]


    # --------------------------------------------------
    # EVALUATION
    # --------------------------------------------------

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


    auc_scores.append(auc)
    f1_scores.append(f1)
    mcc_scores.append(mcc)


    print("\nFold results:")

    print(f"ROC-AUC: {auc:.4f}")
    print(f"F1:      {f1:.4f}")
    print(f"MCC:     {mcc:.4f}")


# --------------------------------------------------
# FINAL RESULTS
# --------------------------------------------------

print("\n========================================")
print("XGBOOST CROSS-VALIDATION RESULTS")
print("========================================")


print("\nROC-AUC scores:")
print([
    round(score, 4)
    for score in auc_scores
])


print("\nF1 scores:")
print([
    round(score, 4)
    for score in f1_scores
])


print("\nMCC scores:")
print([
    round(score, 4)
    for score in mcc_scores
])


# --------------------------------------------------
# MEAN AND STANDARD DEVIATION
# --------------------------------------------------

print("\n----------------------------------------")
print("MEAN ± STANDARD DEVIATION")
print("----------------------------------------")


print(
    f"ROC-AUC: "
    f"{np.mean(auc_scores):.4f} "
    f"± {np.std(auc_scores):.4f}"
)

print(
    f"F1:      "
    f"{np.mean(f1_scores):.4f} "
    f"± {np.std(f1_scores):.4f}"
)

print(
    f"MCC:     "
    f"{np.mean(mcc_scores):.4f} "
    f"± {np.std(mcc_scores):.4f}"
)


print("\n========================================")
print("XGBOOST CROSS-VALIDATION COMPLETE")
print("========================================")

print(
    f"Valid folds evaluated: "
    f"{len(auc_scores)}"
)