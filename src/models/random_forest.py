import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import StratifiedGroupKFold


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

INPUT_FILE = "data/raw/ast_features_labeled.csv"

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded successfully.")
print(f"Number of methods: {len(df)}")


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

groups = df["File"]


# --------------------------------------------------
# 3. Create leakage-safe train/test split
# --------------------------------------------------

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

selected_split = None

for train_indices, test_indices in splitter.split(
        X, y, groups=groups):

    y_train_candidate = y.iloc[train_indices]
    y_test_candidate = y.iloc[test_indices]

    if (
        y_train_candidate.nunique() == 2
        and y_test_candidate.nunique() == 2
    ):
        selected_split = (train_indices, test_indices)
        break


if selected_split is None:
    raise ValueError(
        "Could not find a valid train/test split "
        "containing both classes."
    )


train_indices, test_indices = selected_split


X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]


# --------------------------------------------------
# 4. Display split information
# --------------------------------------------------

print("\n----------------------------------------")
print("TRAIN / TEST DATA")
print("----------------------------------------")

print(f"Training methods: {len(X_train)}")
print(f"Testing methods:  {len(X_test)}")

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())


# --------------------------------------------------
# 5. Create Random Forest model
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)


# --------------------------------------------------
# 6. Train model
# --------------------------------------------------

print("\n----------------------------------------")
print("TRAINING RANDOM FOREST")
print("----------------------------------------")

model.fit(X_train, y_train)

print("Random Forest training complete.")


# --------------------------------------------------
# 7. Make predictions
# --------------------------------------------------

y_pred = model.predict(X_test)

# Probability of the defective class
y_prob = model.predict_proba(X_test)[:, 1]


# --------------------------------------------------
# 8. Evaluate model
# --------------------------------------------------

auc = roc_auc_score(y_test, y_prob)

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
# 9. Display results
# --------------------------------------------------

print("\n----------------------------------------")
print("RANDOM FOREST RESULTS")
print("----------------------------------------")

print(f"ROC-AUC: {auc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"MCC:      {mcc:.4f}")


# --------------------------------------------------
# 10. Confusion matrix
# --------------------------------------------------

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# --------------------------------------------------
# 11. Classification report
# --------------------------------------------------

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# --------------------------------------------------
# 12. Feature importance
# --------------------------------------------------

feature_importance = pd.DataFrame({
    "Feature": AST_FEATURES,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n----------------------------------------")
print("FEATURE IMPORTANCE")
print("----------------------------------------")

print(feature_importance.to_string(index=False))