import pandas as pd
from pathlib import Path


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

AST_FILE = BASE_DIR / "data" / "raw" / "ast_features_labeled.csv"
PROMISE_FILE = BASE_DIR / "data" / "raw" / "ant-1.7.csv"

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "traditional_metrics.csv"
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("Loading AST data...")
ast = pd.read_csv(AST_FILE)

print("Loading PROMISE data...")
promise = pd.read_csv(PROMISE_FILE)


# --------------------------------------------------
# CLEAN COLUMN NAMES
# --------------------------------------------------

ast.columns = ast.columns.str.strip()
promise.columns = promise.columns.str.strip()


# --------------------------------------------------
# REMOVE AST ROWS WITHOUT A CLASS
# --------------------------------------------------

# One AST method has no enclosing class:
# None.java / isSelected
#
# It cannot be matched to a PROMISE class-level label,
# so it is excluded from the traditional-metrics dataset.

before_filter = len(ast)

ast = ast.dropna(subset=["Class"]).copy()

removed_rows = before_filter - len(ast)

print(
    "Removed AST rows without a class:",
    removed_rows
)


# --------------------------------------------------
# CREATE PACKAGE + CLASS FROM PROMISE NAME
# --------------------------------------------------

promise["name"] = (
    promise["name"]
    .astype(str)
    .str.strip()
)

# Everything before the final dot = package
promise["Package"] = (
    promise["name"]
    .str.rsplit(".", n=1)
    .str[0]
)

# Everything after the final dot = class
promise["Class"] = (
    promise["name"]
    .str.rsplit(".", n=1)
    .str[1]
)


# --------------------------------------------------
# CHECK FOR DUPLICATE PACKAGE + CLASS MAPPINGS
# --------------------------------------------------

duplicate_keys = (
    promise
    .groupby(["Package", "Class"])
    .size()
    .reset_index(name="count")
)

duplicates = duplicate_keys[
    duplicate_keys["count"] > 1
]

if not duplicates.empty:

    print(
        "WARNING: Duplicate Package + Class mappings found:"
    )

    print(duplicates)

    raise ValueError(
        "Duplicate Package + Class mappings found "
        "in PROMISE data."
    )


# --------------------------------------------------
# CREATE BINARY DEFECT LABEL
# --------------------------------------------------

promise["bug"] = pd.to_numeric(
    promise["bug"],
    errors="coerce"
)

promise["Defective"] = (
    promise["bug"] > 0
).astype(int)


# --------------------------------------------------
# CHECK REQUIRED AST COLUMNS
# --------------------------------------------------

required_ast_columns = [
    "File",
    "Package",
    "Class",
    "Method",
    "Defective"
]

missing_ast = [
    column
    for column in required_ast_columns
    if column not in ast.columns
]

if missing_ast:

    raise ValueError(
        f"Missing AST columns: {missing_ast}"
    )


# --------------------------------------------------
# TRADITIONAL PROMISE METRICS
# --------------------------------------------------

traditional_features = [
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


# --------------------------------------------------
# CHECK REQUIRED PROMISE COLUMNS
# --------------------------------------------------

required_promise_columns = [
    "Package",
    "Class",
    "bug"
] + traditional_features

missing_promise = [
    column
    for column in required_promise_columns
    if column not in promise.columns
]

if missing_promise:

    raise ValueError(
        f"Missing PROMISE columns: {missing_promise}"
    )


# --------------------------------------------------
# SELECT PROMISE METRICS
# --------------------------------------------------

promise_metrics = promise[
    [
        "Package",
        "Class",
        "bug"
    ] + traditional_features
].copy()


# --------------------------------------------------
# MERGE AST METHODS WITH CLASS-LEVEL METRICS
# --------------------------------------------------

print(
    "Merging AST methods with PROMISE metrics..."
)

merged = ast.merge(
    promise_metrics,
    on=["Package", "Class"],
    how="inner",
    validate="many_to_one"
)


# --------------------------------------------------
# REPORT RESULTS
# --------------------------------------------------

print()
print("Merge complete.")

print(
    "AST method rows:",
    len(ast)
)

print(
    "PROMISE classes:",
    len(promise_metrics)
)

print(
    "Merged method rows:",
    len(merged)
)

print(
    "Matched source files:",
    merged["File"].nunique()
)


# --------------------------------------------------
# CHECK FOR UNMATCHED PACKAGE + CLASS COMBINATIONS
# --------------------------------------------------

ast_keys = set(
    zip(
        ast["Package"],
        ast["Class"]
    )
)

promise_keys = set(
    zip(
        promise_metrics["Package"],
        promise_metrics["Class"]
    )
)

unmatched_keys = ast_keys - promise_keys

print(
    "Unmatched Package + Class combinations:",
    len(unmatched_keys)
)


# --------------------------------------------------
# SAVE DATASET
# --------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

merged.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print(
    "Traditional metrics dataset saved to:"
)

print(OUTPUT_FILE)