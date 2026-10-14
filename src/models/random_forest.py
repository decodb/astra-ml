
import os
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

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
OUTPUT_DIR = "data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

# Remove rows without a usable label, file, or AST feature
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

required_columns = AST_FEATURES + ["File", "Defective"]
df = df.dropna(subset=required_columns).copy()

# Ensure numeric feature values and binary labels
df[AST_FEATURES] = df[AST_FEATURES].apply(
    pd.to_numeric, errors="coerce"
)
df["Defective"] = pd.to_numeric(
    df["Defective"], errors="coerce"
)

df = df.dropna(subset=AST_FEATURES + ["Defective"]).copy()
df = df[df["Defective"].isin([0, 1])].copy()
df["Defective"] = df["Defective"].astype(int)

X = df[AST_FEATURES]
y = df["Defective"]
groups = df["File"]

print("Dataset loaded successfully.")
print(f"Number of methods: {len(df)}")
print(f"Number of files: {df['File'].nunique()}")

print("\nClass distribution:")
print(y.value_counts().sort_index())
print("\nClass percentages:")
print((y.value_counts(normalize=True).sort_index() * 100).round(2))


# --------------------------------------------------
# 2. Create file-level cross-validation
# --------------------------------------------------

N_SPLITS = 5

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=42
)


# --------------------------------------------------
# 3. Store evaluation and SHAP results
# --------------------------------------------------

auc_scores = []
f1_scores = []
mcc_scores = []

shap_values_all = []
X_explained_all = []


# --------------------------------------------------
# 4. Cross-validation
# --------------------------------------------------

print("\n========================================")
print("FILE-LEVEL CROSS-VALIDATION")
print("========================================")

for fold, (train_indices, test_indices) in enumerate(
    cv.split(X, y, groups=groups), start=1
):

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]

    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    train_files = set(groups.iloc[train_indices])
    test_files = set(groups.iloc[test_indices])

    overlap = train_files.intersection(test_files)

    print(f"\n--------------- FOLD {fold} ---------------")
    print(f"Training methods: {len(X_train)}")
    print(f"Testing methods:  {len(X_test)}")
    print(f"Training files:   {len(train_files)}")
    print(f"Testing files:    {len(test_files)}")

    if overlap:
        raise RuntimeError(
            f"File-level leakage detected in fold {fold}: {overlap}"
        )

    print("Leakage check: PASS")

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        print("Skipping fold: only one class is present.")
        continue

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluation
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    auc_scores.append(auc)
    f1_scores.append(f1)
    mcc_scores.append(mcc)

    print(f"ROC-AUC: {auc:.4f}")
    print(f"F1:      {f1:.4f}")
    print(f"MCC:     {mcc:.4f}")

    # ----------------------------------------------
    # SHAP explanations for held-out test methods
    # ----------------------------------------------

    explainer = shap.TreeExplainer(model)
    shap_output = explainer.shap_values(X_test)

    # Handle SHAP's different output formats
    if isinstance(shap_output, list):
        # Binary classifier: class 1 = defective
        shap_values = np.asarray(shap_output[1])
    elif hasattr(shap_output, "values"):
        shap_values = np.asarray(shap_output.values)
    else:
        shap_values = np.asarray(shap_output)

    # Some SHAP versions return:
    # (samples, features, classes)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    if shap_values.shape != X_test.shape:
        raise ValueError(
            f"Unexpected SHAP shape {shap_values.shape}; "
            f"expected {X_test.shape}"
        )

    shap_values_all.append(shap_values)
    X_explained_all.append(X_test.copy())

    print("SHAP values calculated for held-out methods.")


# --------------------------------------------------
# 5. Display overall model performance
# --------------------------------------------------

print("\n========================================")
print("CROSS-VALIDATION RESULTS")
print("========================================")

if not auc_scores:
    raise ValueError("No valid folds were available.")

print(f"Valid folds: {len(auc_scores)}")

print(
    f"ROC-AUC: {np.mean(auc_scores):.4f} "
    f"+/- {np.std(auc_scores):.4f}"
)
print(
    f"F1:      {np.mean(f1_scores):.4f} "
    f"+/- {np.std(f1_scores):.4f}"
)
print(
    f"MCC:     {np.mean(mcc_scores):.4f} "
    f"+/- {np.std(mcc_scores):.4f}"
)


# --------------------------------------------------
# 6. RQ2: Aggregate SHAP results
# --------------------------------------------------

print("\n========================================")
print("RQ2: SHAP FEATURE IMPORTANCE")
print("========================================")

all_shap_values = np.vstack(shap_values_all)
all_X_explained = pd.concat(
    X_explained_all,
    ignore_index=True
)

# Mean absolute SHAP: overall importance
mean_abs_shap = np.abs(all_shap_values).mean(axis=0)

# Mean signed SHAP: average direction of contribution
mean_signed_shap = all_shap_values.mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": AST_FEATURES,
    "Mean_Absolute_SHAP": mean_abs_shap,
    "Mean_Signed_SHAP": mean_signed_shap
}).sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
)

importance_path = os.path.join(
    OUTPUT_DIR, "ast_shap_importance.csv"
)
importance_df.to_csv(importance_path, index=False)

print("\nAST features ranked by SHAP importance:")
print(importance_df.to_string(index=False))

print(f"\nSaved feature rankings to: {importance_path}")


# --------------------------------------------------
# 7. SHAP importance bar plot
# --------------------------------------------------

plt.figure()

shap.summary_plot(
    all_shap_values,
    all_X_explained,
    plot_type="bar",
    show=False
)

plt.tight_layout()

bar_path = os.path.join(
    OUTPUT_DIR, "ast_shap_importance.png"
)

plt.savefig(
    bar_path,
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print(f"Saved importance plot to: {bar_path}")


# --------------------------------------------------
# 8. SHAP summary plot
# --------------------------------------------------

plt.figure()

shap.summary_plot(
    all_shap_values,
    all_X_explained,
    show=False
)

plt.tight_layout()

summary_path = os.path.join(
    OUTPUT_DIR, "ast_shap_summary.png"
)

plt.savefig(
    summary_path,
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print(f"Saved SHAP summary plot to: {summary_path}")


# --------------------------------------------------
# 9. Final summary
# --------------------------------------------------

print("\n========================================")
print("ANALYSIS COMPLETE")
print("========================================")
print("RQ2 outputs:")
print("- ast_shap_importance.csv")
print("- ast_shap_importance.png")
print("- ast_shap_summary.png")
