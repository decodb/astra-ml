import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

INPUT_FILE = "data/raw/ast_features_labeled.csv"


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded successfully.")
print(f"Number of rows: {len(df)}")
print(f"Number of columns: {len(df.columns)}")


# --------------------------------------------------
# 3. Display basic information
# --------------------------------------------------

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# --------------------------------------------------
# 4. Check for missing values
# --------------------------------------------------

print("\nMissing values:")
print(df.isnull().sum())


# --------------------------------------------------
# 5. Check target distribution
# --------------------------------------------------

print("\nDefective class distribution:")
print(df["Defective"].value_counts())

print("\nDefective class percentages:")
print(df["Defective"].value_counts(normalize=True) * 100)


# --------------------------------------------------
# 6. Select AST features
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


# --------------------------------------------------
# 7. Display feature information
# --------------------------------------------------

print("\nAST features:")
print(AST_FEATURES)

print("\nFeature matrix shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)


# --------------------------------------------------
# 8. Split data by file
# --------------------------------------------------
# Methods from the same Java file must stay together.
# This prevents methods from the same file appearing
# in both training and testing data.
#
# StratifiedGroupKFold also tries to preserve the
# defective/non-defective class distribution.

groups = df["File"]

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# Find a split where the test set contains both classes
selected_split = None

for train_indices, test_indices in splitter.split(
        X, y, groups=groups):

    y_train_candidate = y.iloc[train_indices]
    y_test_candidate = y.iloc[test_indices]

    # We need both classes in both sets
    if (
        y_train_candidate.nunique() == 2
        and y_test_candidate.nunique() == 2
    ):
        selected_split = (train_indices, test_indices)
        break


# Make sure a suitable split was found
if selected_split is None:
    raise ValueError(
        "Could not find a train/test split containing "
        "both defective and non-defective classes. "
        "More Java files may be needed."
    )


train_indices, test_indices = selected_split


X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]


# --------------------------------------------------
# 9. Display split information
# --------------------------------------------------

print("\n----------------------------------------")
print("TRAIN / TEST SPLIT")
print("----------------------------------------")

print(f"Training methods: {len(X_train)}")
print(f"Testing methods:  {len(X_test)}")

print("\nTraining files:")
print(df.iloc[train_indices]["File"].unique())

print("\nTesting files:")
print(df.iloc[test_indices]["File"].unique())


# --------------------------------------------------
# 10. Check target distribution after split
# --------------------------------------------------

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())


print("\nTraining class percentages:")
print(y_train.value_counts(normalize=True) * 100)

print("\nTesting class percentages:")
print(y_test.value_counts(normalize=True) * 100)


# --------------------------------------------------
# 11. Verify that files do not overlap
# --------------------------------------------------

train_files = set(
    df.iloc[train_indices]["File"]
)

test_files = set(
    df.iloc[test_indices]["File"]
)

overlap = train_files.intersection(test_files)


print("\n----------------------------------------")
print("LEAKAGE CHECK")
print("----------------------------------------")

if len(overlap) == 0:
    print(
        "PASS: No files appear in both "
        "training and testing sets."
    )
else:
    print("WARNING: File overlap detected!")
    print(overlap)