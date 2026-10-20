
from pathlib import Path
import pandas as pd
import numpy as np

# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

AST_FILE = ROOT / "data/raw/ast_features_labeled.csv"
PROMISE_FILE = ROOT / "data/raw/ant-1.7.csv"
TRADITIONAL_FILE = ROOT / "data/processed/traditional_metrics.csv"

AST_FEATURES = [
    "If", "For", "While", "Return", "TryCatch",
    "MethodCalls", "Variables", "Parameters",
    "ASTDepth", "BranchingFactor", "LOC"
]

LABEL_COLUMNS = ["bug", "Defective"]


def heading(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


def normalize(series):
    return series.astype("string").str.strip()


# --------------------------------------------------
# 2. Load datasets
# --------------------------------------------------

heading("LOADING DATASETS")

ast = pd.read_csv(AST_FILE)
promise = pd.read_csv(PROMISE_FILE)

ast.columns = ast.columns.str.strip()
promise.columns = promise.columns.str.strip()

print(f"AST rows: {len(ast)}")
print(f"AST files: {ast['File'].nunique()}")
print(f"PROMISE class records: {len(promise)}")

required_ast = [
    "File", "Package", "Class", "Method",
    *AST_FEATURES, *LABEL_COLUMNS
]

missing_columns = set(required_ast) - set(ast.columns)

if missing_columns:
    raise ValueError(f"AST dataset missing columns: {missing_columns}")

if not {"name", "bug"}.issubset(promise.columns):
    raise ValueError("PROMISE CSV must contain 'name' and 'bug'.")


# --------------------------------------------------
# 3. Missing values and duplicate methods
# --------------------------------------------------

heading("CHECK 1: MISSING VALUES")

print("\nAST missing values:")
print(ast[required_ast].isna().sum().to_string())

print("\nRows with missing file/class/method:")
bad_identity = ast[
    ast[["File", "Class", "Method"]].isna().any(axis=1)
]
print(len(bad_identity))

print("\nRows with missing AST features:")
print(ast[AST_FEATURES].isna().any(axis=1).sum())

heading("CHECK 2: DUPLICATE METHOD RECORDS")

# Exact duplicate rows
print("Exact duplicate rows:", ast.duplicated().sum())

# Duplicate method signatures within a source file
identity_cols = ["File", "Class", "Method"]
valid_identity = ast.dropna(subset=identity_cols)

duplicate_methods = valid_identity[
    valid_identity.duplicated(identity_cols, keep=False)
]

print("Rows involved in repeated File/Class/Method keys:",
      len(duplicate_methods))

if not duplicate_methods.empty:
    print(duplicate_methods[identity_cols].head(20).to_string(index=False))


# --------------------------------------------------
# 4. Feature validity
# --------------------------------------------------

heading("CHECK 3: AST FEATURE VALIDITY")

for col in AST_FEATURES:
    ast[col] = pd.to_numeric(ast[col], errors="coerce")

print("Non-numeric or missing feature values:")
print(ast[AST_FEATURES].isna().sum().to_string())

print("\nNegative feature values:")
for col in AST_FEATURES:
    count = (ast[col] < 0).sum()
    if count:
        print(f"{col}: {count}")

print("\nInfinite feature values:")
print(np.isinf(ast[AST_FEATURES].to_numpy(dtype=float)).sum())

print("\nBranchingFactor summary:")
print(ast["BranchingFactor"].describe().to_string())


# --------------------------------------------------
# 5. Label validity
# --------------------------------------------------

heading("CHECK 4: LABEL VALIDITY")

ast["bug"] = pd.to_numeric(ast["bug"], errors="coerce")
ast["Defective"] = pd.to_numeric(ast["Defective"], errors="coerce")
promise["bug"] = pd.to_numeric(promise["bug"], errors="coerce")

print("Missing bug labels:", ast["bug"].isna().sum())
print("Missing Defective labels:", ast["Defective"].isna().sum())

print("\nDefective label values:")
print(ast["Defective"].value_counts(dropna=False).to_string())

invalid_labels = ast[
    ~ast["Defective"].isin([0, 1])
]
print("\nRows with labels other than 0/1:", len(invalid_labels))

# Check that Defective agrees with bug > 0
valid_bug = ast["bug"].notna()
expected_label = (ast.loc[valid_bug, "bug"] > 0).astype(int)

mismatch = (
    ast.loc[valid_bug, "Defective"].astype("Int64")
    != expected_label
)

print("Rows where Defective disagrees with bug > 0:",
      int(mismatch.sum()))

print("\nBug count summary:")
print(ast["bug"].describe().to_string())

print("\nNegative bug counts:", (ast["bug"] < 0).sum())


# --------------------------------------------------
# 6. Parse PROMISE package/class keys
# --------------------------------------------------

heading("CHECK 5: PROMISE CLASS MAPPING")

promise["name"] = normalize(promise["name"])
ast["Package"] = normalize(ast["Package"])
ast["Class"] = normalize(ast["Class"])

def parse_promise_name(name):
    if pd.isna(name) or not name:
        return pd.Series([pd.NA, pd.NA])

    # PROMISE class name is expected to be package.Class
    package, separator, class_name = name.rpartition(".")

    if not separator:
        return pd.Series([pd.NA, name])

    return pd.Series([package, class_name])


promise[["Package_key", "Class_key"]] = promise["name"].apply(
    parse_promise_name
)

promise["Package_key"] = normalize(promise["Package_key"])
promise["Class_key"] = normalize(promise["Class_key"])

promise_key = ["Package_key", "Class_key"]
ast_key = ["Package", "Class"]

print("PROMISE rows missing parsed package/class:",
      promise[promise[promise_key].isna().any(axis=1)].shape[0])

duplicate_classes = promise[
    promise.duplicated(promise_key, keep=False)
    & promise[promise_key].notna().all(axis=1)
]

print("PROMISE rows involved in duplicate package/class keys:",
      len(duplicate_classes))

# Only safely compare unique class keys
promise_valid = promise.dropna(subset=promise_key + ["bug"]).copy()

duplicate_key_mask = promise_valid.duplicated(
    promise_key, keep=False
)

promise_unique = promise_valid.loc[~duplicate_key_mask].copy()

ast_keys = ast[ast_key].drop_duplicates()
promise_keys = promise_unique[promise_key].drop_duplicates()

ast_key_set = set(map(tuple, ast_keys.to_numpy()))
promise_key_set = set(map(tuple, promise_keys.to_numpy()))

print("Unique AST package/class pairs:", len(ast_key_set))
print("Unique unambiguous PROMISE package/class pairs:",
      len(promise_key_set))
print("AST class pairs with no unique PROMISE match:",
      len(ast_key_set - promise_key_set))
print("PROMISE class pairs not represented in AST:",
      len(promise_key_set - ast_key_set))

# Verify labels against original PROMISE bug counts
lookup = promise_unique.set_index(promise_key)["bug"]

ast["promise_bug_check"] = [
    lookup.get((pkg, cls), np.nan)
    for pkg, cls in zip(ast["Package"], ast["Class"])
]

matched = ast["promise_bug_check"].notna()
print("\nAST method rows with an unambiguous PROMISE match:",
      int(matched.sum()))
print("AST method rows without an unambiguous PROMISE match:",
      int((~matched).sum()))

bug_mismatch = matched & (
    ast["bug"].isna()
    | ~np.isclose(
        ast["bug"].fillna(-999).astype(float),
        ast["promise_bug_check"].fillna(-999).astype(float)
    )
)

print("Rows whose bug count differs from PROMISE:",
      int(bug_mismatch.sum()))

if bug_mismatch.any():
    print(ast.loc[
        bug_mismatch,
        ["File", "Package", "Class", "Method",
         "bug", "promise_bug_check", "Defective"]
    ].head(20).to_string(index=False))


# --------------------------------------------------
# 7. Label consistency within classes and files
# --------------------------------------------------

heading("CHECK 6: LABEL CONSISTENCY")

class_labels = ast.dropna(subset=["Package", "Class"]).groupby(
    ["Package", "Class"]
)["Defective"].nunique()

file_labels = ast.dropna(subset=["File"]).groupby(
    "File"
)["Defective"].nunique()

print("Classes containing multiple Defective labels:",
      int((class_labels > 1).sum()))

print("Files containing multiple Defective labels:",
      int((file_labels > 1).sum()))

print("\nMethods per file:")
print(ast.groupby("File").size().describe().to_string())


# --------------------------------------------------
# 8. Check the traditional-metrics dataset
# --------------------------------------------------

heading("CHECK 7: TRADITIONAL DATASET")

if TRADITIONAL_FILE.exists():
    traditional = pd.read_csv(TRADITIONAL_FILE)
    traditional.columns = traditional.columns.str.strip()

    print("Traditional rows:", len(traditional))
    print("Traditional files:", traditional["File"].nunique()
          if "File" in traditional.columns else "File column absent")

    traditional_required = AST_FEATURES + [
        "File", "Package", "Class", "Defective"
    ]
    missing = set(traditional_required) - set(traditional.columns)
    print("Missing expected columns:", sorted(missing))

    if "Defective" in traditional.columns:
        print("\nTraditional label distribution:")
        print(traditional["Defective"].value_counts(dropna=False).to_string())

    if {"Package", "Class", "bug", "Defective"}.issubset(
        traditional.columns
    ):
        traditional["bug"] = pd.to_numeric(
            traditional["bug"], errors="coerce"
        )
        traditional["Defective"] = pd.to_numeric(
            traditional["Defective"], errors="coerce"
        )

        valid = traditional["bug"].notna() & traditional["Defective"].notna()
        inconsistent = valid & (
            traditional["Defective"] !=
            (traditional["bug"] > 0).astype(int)
        )

        print("Traditional rows with inconsistent bug labels:",
              int(inconsistent.sum()))

    if {"File", "Class", "Method"}.issubset(traditional.columns):
        print("Duplicate traditional method keys:",
              traditional.duplicated(
                  ["File", "Class", "Method"]
              ).sum())
else:
    print("Traditional dataset not found; skipped.")


# --------------------------------------------------
# 9. Final summary
# --------------------------------------------------

heading("AUDIT COMPLETE")

print("Review every warning before retraining models.")
print("Remember: Defective is inherited from class-level labels.")
print("This audit does not prove individual methods are defective.")
