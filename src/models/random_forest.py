import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    matthews_corrcoef
)


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

INPUT_FILE = "data/raw/ast_features_labeled.csv"

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded successfully.")
print(f"Number of methods: {len(df)}")
print(f"Number of files: {df['File'].nunique()}")


# --------------------------------------------------
# 2. Select AST features
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

# Group = Java file
groups = df["File"]


# --------------------------------------------------
# 3. Display dataset information
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
# 4. Create file-level cross-validation
# --------------------------------------------------

N_SPLITS = 5

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=42
)


# --------------------------------------------------
# 5. Store results
# --------------------------------------------------

auc_scores = []
f1_scores = []
mcc_scores = []


# --------------------------------------------------
# 6. Cross-validation
# --------------------------------------------------

print("\n========================================")
print("FILE-LEVEL CROSS-VALIDATION")
print("========================================")


for fold, (train_indices, test_indices) in enumerate(
        cv.split(X, y, groups=groups), start=1):

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]

    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    train_files = set(
        df.iloc[train_indices]["File"]
    )

    test_files = set(
        df.iloc[test_indices]["File"]
    )

    overlap = train_files.intersection(test_files)


    print(f"\n----------------------------------------")
    print(f"FOLD {fold}")
    print(f"----------------------------------------")

    print(f"Training methods: {len(X_train)}")
    print(f"Testing methods:  {len(X_test)}")

    print(f"Training files: {len(train_files)}")
    print(f"Testing files:  {len(test_files)}")

    print("\nTraining class distribution:")
    print(y_train.value_counts().to_dict())

    print("\nTesting class distribution:")
    print(y_test.value_counts().to_dict())


    # --------------------------------------------------
    # Leakage check
    # --------------------------------------------------

    if len(overlap) == 0:
        print("\nLeakage check: PASS")
    else:
        print("\nLeakage check: FAIL")
        print("Overlapping files:", overlap)


    # --------------------------------------------------
    # Check whether both classes exist
    # --------------------------------------------------

    if y_train.nunique() < 2:
        print("Skipping fold: training set contains only one class.")
        continue

    if y_test.nunique() < 2:
        print("Skipping fold: testing set contains only one class.")
        continue


    # --------------------------------------------------
    # Create Random Forest
    # --------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )


    # --------------------------------------------------
    # Train
    # --------------------------------------------------

    model.fit(X_train, y_train)


    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]


    # --------------------------------------------------
    # Evaluation
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


    # --------------------------------------------------
    # Store results
    # --------------------------------------------------

    auc_scores.append(auc)
    f1_scores.append(f1)
    mcc_scores.append(mcc)


    # --------------------------------------------------
    # Display fold results
    # --------------------------------------------------

    print("\nFold results:")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"F1:      {f1:.4f}")
    print(f"MCC:     {mcc:.4f}")


# --------------------------------------------------
# 7. Calculate overall results
# --------------------------------------------------

print("\n========================================")
print("CROSS-VALIDATION RESULTS")
print("========================================")


if len(auc_scores) == 0:
    raise ValueError(
        "No valid folds were available for evaluation."
    )


print("\nROC-AUC scores:")
print([round(score, 4) for score in auc_scores])

print("\nF1 scores:")
print([round(score, 4) for score in f1_scores])

print("\nMCC scores:")
print([round(score, 4) for score in mcc_scores])


# --------------------------------------------------
# 8. Mean and standard deviation
# --------------------------------------------------

print("\n----------------------------------------")
print("MEAN ± STANDARD DEVIATION")
print("----------------------------------------")

print(
    f"ROC-AUC: {np.mean(auc_scores):.4f} "
    f"± {np.std(auc_scores):.4f}"
)

print(
    f"F1:      {np.mean(f1_scores):.4f} "
    f"± {np.std(f1_scores):.4f}"
)

print(
    f"MCC:     {np.mean(mcc_scores):.4f} "
    f"± {np.std(mcc_scores):.4f}"
)


# --------------------------------------------------
# 9. Final summary
# --------------------------------------------------

print("\n========================================")
print("RANDOM FOREST CROSS-VALIDATION COMPLETE")
print("========================================")

print(f"Valid folds evaluated: {len(auc_scores)}")