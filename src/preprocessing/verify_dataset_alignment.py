import pandas as pd

original_path = "data/raw/ast_features_labeled.csv"
diagnostic_path = "data/raw/ast_features_with_line_numbers.csv"

original = pd.read_csv(original_path)
diagnostic = pd.read_csv(diagnostic_path)

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
    "LOC",
]

MATCH_COLUMNS = [
    "File",
    "Class",
    "Method",
] + AST_FEATURES

print("=" * 65)
print("DATASET ALIGNMENT VERIFICATION")
print("=" * 65)

print("\nOriginal dataset:")
print("Rows:", len(original))
print("Columns:", len(original.columns))

print("\nDiagnostic dataset:")
print("Rows:", len(diagnostic))
print("Columns:", len(diagnostic.columns))

missing_original = set(MATCH_COLUMNS) - set(original.columns)
missing_diagnostic = set(MATCH_COLUMNS) - set(diagnostic.columns)

if missing_original or missing_diagnostic:
    raise ValueError(
        f"Missing columns - original: {missing_original}, "
        f"diagnostic: {missing_diagnostic}"
    )

original_counts = (
    original.groupby(MATCH_COLUMNS, dropna=False)
    .size()
    .rename("OriginalCount")
    .reset_index()
)

diagnostic_counts = (
    diagnostic.groupby(MATCH_COLUMNS, dropna=False)
    .size()
    .rename("DiagnosticCount")
    .reset_index()
)

comparison = original_counts.merge(
    diagnostic_counts,
    on=MATCH_COLUMNS,
    how="outer",
)

comparison["OriginalCount"] = (
    comparison["OriginalCount"].fillna(0).astype(int)
)

comparison["DiagnosticCount"] = (
    comparison["DiagnosticCount"].fillna(0).astype(int)
)

differences = comparison[
    comparison["OriginalCount"] != comparison["DiagnosticCount"]
]

print("\nALIGNMENT RESULTS")
print("Original unique feature records:", len(original_counts))
print("Diagnostic unique feature records:", len(diagnostic_counts))
print("Mismatched feature groups:", len(differences))

if len(original) != len(diagnostic):
    print("\nWARNING: Row counts differ.")

if differences.empty and len(original) == len(diagnostic):
    print("\nPASS: Both datasets contain the same feature records.")
else:
    print("\nWARNING: Dataset differences detected.")
    print(differences.head(20).to_string(index=False))

print("\nNo datasets were modified.")