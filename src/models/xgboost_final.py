
import pandas as pd
import numpy as np

from pathlib import Path

from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)

# ============================================================
# STEP 1: LOAD FINAL VERIFIED DATASET
# ============================================================

DATA_PATH = "data/raw/ast_features_final_labeled.csv"

df = pd.read_csv(
    DATA_PATH,
    keep_default_na=False
)

FEATURES = [
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

X = df[FEATURES].astype(float)
y = df["Defective"].astype(int)

# Group methods from the same Java source file together.
groups = (
    df["Package"].astype(str)
    + "/"
    + df["File"].astype(str)
)

print("=" * 65)
print("XGBOOST - FINAL DATASET")
print("=" * 65)

print("Methods:", len(df))
print("Features:", len(FEATURES))
print("Source-file groups:", groups.nunique())

print("\nClass distribution:")
print(y.value_counts().sort_index())

# ============================================================
# STEP 2: CONFIGURE GROUPED CROSS-VALIDATION
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []
predictions = []

# ============================================================
# STEP 3: TRAIN AND EVALUATE XGBOOST
# ============================================================

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups),
    start=1
):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])

    # Prevent methods from the same file appearing
    # in both training and testing.
    assert train_groups.isdisjoint(test_groups), (
        "Data leakage: source files overlap."
    )

    # Balance the classes using training data only.
    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    if positive_count == 0:
        raise ValueError(
            f"Fold {fold} has no positive training examples."
        )

    scale_pos_weight = negative_count / positive_count

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Predictions and probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Fold": fold,
        "Accuracy": accuracy_score(
            y_test, y_pred
        ),
        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "F1": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "ROC_AUC": (
            roc_auc_score(y_test, y_prob)
            if y_test.nunique() == 2
            else np.nan
        ),
        "MCC": matthews_corrcoef(
            y_test, y_pred
        )
    }

    results.append(metrics)

    fold_predictions = pd.DataFrame({
        "Fold": fold,
        "FileGroup": groups.iloc[test_idx].to_numpy(),
        "Actual": y_test.to_numpy(),
        "Predicted": y_pred,
        "Probability": y_prob
    })

    predictions.append(fold_predictions)

    print(f"\nFOLD {fold}")
    print("Training methods:", len(train_idx))
    print("Testing methods:", len(test_idx))
    print("Training files:", len(train_groups))
    print("Testing files:", len(test_groups))

    print("F1-score:", round(metrics["F1"], 4))
    print("ROC-AUC:", round(metrics["ROC_AUC"], 4))
    print("MCC:", round(metrics["MCC"], 4))

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1]
        )
    )

# ============================================================
# STEP 4: SUMMARIZE CROSS-VALIDATION RESULTS
# ============================================================

results_df = pd.DataFrame(results)

predictions_df = pd.concat(
    predictions,
    ignore_index=True
)

print("\n" + "=" * 65)
print("5-FOLD CROSS-VALIDATION SUMMARY")
print("=" * 65)

print("\nMean metrics:")
print(
    results_df.drop(columns=["Fold"])
    .mean()
    .round(4)
)

print("\nStandard deviation:")
print(
    results_df.drop(columns=["Fold"])
    .std()
    .round(4)
)

# ============================================================
# STEP 5: SAVE RESULTS
# ============================================================

output_dir = Path("results/xgboost_final")
output_dir.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    output_dir / "fold_metrics.csv",
    index=False
)

predictions_df.to_csv(
    output_dir / "predictions.csv",
    index=False
)

print("\nResults saved to:", output_dir)
print("Original datasets were not modified.")
