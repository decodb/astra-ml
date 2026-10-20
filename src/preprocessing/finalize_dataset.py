
import pandas as pd
from pathlib import Path

ORIGINAL_PATH = Path("data/raw/ast_features_labeled.csv")
DIAGNOSTIC_PATH = Path(
    "data/raw/ast_features_with_line_numbers.csv"
)
PROMISE_PATH = Path("data/raw/ant-1.7.csv")
OUTPUT_PATH = Path(
    "data/raw/ast_features_final_labeled.csv"
)

AST_FEATURES = [
    "If", "For", "While", "Return", "TryCatch",
    "MethodCalls", "Variables", "Parameters",
    "ASTDepth", "BranchingFactor", "LOC"
]

MATCH_COLUMNS = ["File", "Class", "Method"] + AST_FEATURES

# Load data, preserving class names such as "None".
original = pd.read_csv(
    ORIGINAL_PATH, keep_default_na=False
)
diagnostic = pd.read_csv(
    DIAGNOSTIC_PATH, keep_default_na=False
)
promise = pd.read_csv(
    PROMISE_PATH, keep_default_na=False
)

# Ensure the source datasets are aligned.
if len(original) != len(diagnostic):
    raise ValueError("Dataset row counts differ.")

if not original[MATCH_COLUMNS].equals(
    diagnostic[MATCH_COLUMNS]
):
    raise ValueError("Dataset row order or features differ.")

# Check that PROMISE names are unique.
if promise["name"].duplicated().any():
    raise ValueError("Duplicate PROMISE class names found.")

# Reconstruct qualified class names.
qualified_class = (
    diagnostic["Package"].astype(str)
    + "."
    + diagnostic["Class"].astype(str)
).str.strip(".")

# Verify against PROMISE.
promise_bugs = qualified_class.map(
    promise.set_index("name")["bug"]
)

if promise_bugs.isna().any():
    raise ValueError("Unmatched PROMISE classes found.")

if not (
    original["bug"].to_numpy()
    == promise_bugs.to_numpy()
).all():
    raise ValueError("PROMISE bug counts do not match.")

expected_labels = promise_bugs.gt(0).astype(int)

if not (
    original["Defective"].to_numpy()
    == expected_labels.to_numpy()
).all():
    raise ValueError("Binary defect labels do not match.")

# Combine diagnostic metadata and verified labels.
final = diagnostic.copy()
final["bug"] = promise_bugs.to_numpy()
final["Defective"] = expected_labels.to_numpy()

# Preserve all extracted methods.
assert len(final) == 8064

# Protect any previously created final dataset.
if OUTPUT_PATH.exists():
    raise FileExistsError(
        f"{OUTPUT_PATH} already exists. "
        "Review it before replacing."
    )

OUTPUT_PATH.parent.mkdir(
    parents=True, exist_ok=True
)

final.to_csv(OUTPUT_PATH, index=False)

# Verify the written CSV.
saved = pd.read_csv(
    OUTPUT_PATH, keep_default_na=False
)

assert len(saved) == len(final)
assert saved.columns.tolist() == final.columns.tolist()

print("=" * 60)
print("FINAL DATASET CREATED SUCCESSFULLY")
print("=" * 60)
print("Output:", OUTPUT_PATH)
print("Methods:", len(saved))
print("Columns:", len(saved.columns))
print("Unique source files:", saved["File"].nunique())
print("\nDefect distribution:")
print(saved["Defective"].value_counts().sort_index())
print("\nOriginal datasets remain unchanged.")
