
import pandas as pd
from pathlib import Path

ORIGINAL_PATH = "data/raw/ast_features_labeled.csv"
DIAGNOSTIC_PATH = "data/raw/ast_features_with_line_numbers.csv"
PROMISE_PATH = "data/raw/ant-1.7.csv"

# ============================================================
# STEP 1: LOAD DATASETS
# ============================================================

original = pd.read_csv(
    ORIGINAL_PATH,
    keep_default_na=False
)

diagnostic = pd.read_csv(
    DIAGNOSTIC_PATH,
    keep_default_na=False
)

promise = pd.read_csv(
    PROMISE_PATH,
    keep_default_na=False
)

AST_FEATURES = [
    "If", "For", "While", "Return", "TryCatch",
    "MethodCalls", "Variables", "Parameters",
    "ASTDepth", "BranchingFactor", "LOC"
]

MATCH_COLUMNS = ["File", "Class", "Method"] + AST_FEATURES

print("=" * 65)
print("PROMISE LABEL MAPPING AUDIT")
print("=" * 65)

print("\nDATASET SIZES")
print("Original methods:", len(original))
print("Diagnostic methods:", len(diagnostic))
print("PROMISE classes:", len(promise))

# ============================================================
# STEP 2: VERIFY ROW ALIGNMENT
# ============================================================

same_order = (
    len(original) == len(diagnostic)
    and original[MATCH_COLUMNS]
        .reset_index(drop=True)
        .equals(
            diagnostic[MATCH_COLUMNS].reset_index(drop=True)
        )
)

print("\nROW ALIGNMENT")
print("Identical row order and features:", same_order)

if not same_order:
    print("\nSTOP: Row order differs.")
    print("Labels cannot safely be transferred by row position.")
    print("Original datasets were not modified.")
    raise SystemExit(1)

# ============================================================
# STEP 3: COMBINE METADATA AND EXISTING LABELS
# ============================================================

combined = diagnostic.copy()

combined["OriginalBug"] = original["bug"].to_numpy()
combined["OriginalDefective"] = original["Defective"].to_numpy()

# ============================================================
# STEP 4: RECONSTRUCT FULLY QUALIFIED CLASS NAMES
# ============================================================

# keep_default_na=False preserves Java class names such as "None".

combined["QualifiedClass"] = (
    combined["Package"].astype(str)
    + "."
    + combined["Class"].astype(str)
).str.strip(".")

print("\nCLASS NAME CHECK")
print("Missing class names:",
      (combined["Class"].astype(str).str.strip() == "").sum())

# ============================================================
# STEP 5: CHECK PROMISE CLASS IDENTITIES
# ============================================================

duplicate_promise = promise[
    promise.duplicated(subset=["name"], keep=False)
]

print("\nPROMISE CLASS IDENTITIES")
print("Duplicate PROMISE class names:", len(duplicate_promise))

if not duplicate_promise.empty:
    print("\nSTOP: Ambiguous PROMISE class identities.")
    raise SystemExit(1)

# ============================================================
# STEP 6: MAP ORIGINAL PROMISE BUG COUNTS
# ============================================================

promise_lookup = promise.set_index("name")["bug"]

combined["PromiseBug"] = (
    combined["QualifiedClass"].map(promise_lookup)
)

combined["PromiseDefective"] = (
    combined["PromiseBug"].gt(0)
)

print("\nCLASS MATCHING")
print("Matched methods:", combined["PromiseBug"].notna().sum())
print("Unmatched methods:", combined["PromiseBug"].isna().sum())

# ============================================================
# STEP 7: VERIFY EXISTING LABELS
# ============================================================

matched = combined[
    combined["PromiseBug"].notna()
].copy()

bug_mismatch = matched[
    matched["OriginalBug"] != matched["PromiseBug"]
]

label_mismatch = matched[
    matched["OriginalDefective"]
    != matched["PromiseDefective"].astype(int)
]

print("\nLABEL COMPARISON")
print("Bug-count mismatches:", len(bug_mismatch))
print("Binary-label mismatches:", len(label_mismatch))

# ============================================================
# STEP 8: VERIFY CLASS DISTRIBUTION
# ============================================================

print("\nCLASS DISTRIBUTION")

print("\nOriginal labels:")
print(
    combined["OriginalDefective"]
    .value_counts()
    .sort_index()
)

print("\nPROMISE labels for matched methods:")
print(
    matched["PromiseDefective"]
    .astype(int)
    .value_counts()
    .sort_index()
)

# ============================================================
# STEP 9: SAVE DIAGNOSTIC REPORTS
# ============================================================

output_dir = Path(
    "data/processed/duplicate_diagnostics"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

unmatched = combined[
    combined["PromiseBug"].isna()
]

unmatched.to_csv(
    output_dir / "promise_unmatched_methods.csv",
    index=False
)

bug_mismatch.to_csv(
    output_dir / "promise_bug_mismatches.csv",
    index=False
)

label_mismatch.to_csv(
    output_dir / "promise_label_mismatches.csv",
    index=False
)

# ============================================================
# STEP 10: FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 65)
print("FINAL VERIFICATION RESULT")
print("=" * 65)

if (
    same_order
    and unmatched.empty
    and bug_mismatch.empty
    and label_mismatch.empty
):
    print("PASS: All method labels match PROMISE.")
    print("All", len(combined), "methods verified successfully.")
    print("Dataset is ready for finalization.")

else:
    print("REVIEW REQUIRED: Mapping differences detected.")
    print("Check diagnostic reports before finalizing.")

print("\nOriginal datasets were not modified.")
