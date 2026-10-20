import pandas as pd

DATASET = "data/raw/ast_features_labeled.csv"

df = pd.read_csv(DATASET)

print("=" * 65)
print("PROMISE DEFECT LABEL VERIFICATION")
print("=" * 65)

print("\nDATASET SUMMARY")
print("Total methods:", len(df))
print("Unique source files:", df["File"].nunique())

print("\nDEFECT LABEL DISTRIBUTION")
print(df["Defective"].value_counts(dropna=False))

print("\nMISSING LABELS")
print("Missing bug values:", df["bug"].isna().sum())
print("Missing Defective values:", df["Defective"].isna().sum())

print("\nBINARY LABEL VALIDATION")

invalid_labels = df[
    ~df["Defective"].isin([0, 1])
]

print("Invalid binary labels:", len(invalid_labels))

print("\nBUG-TO-DEFECTIVE CONSISTENCY")

expected_labels = (df["bug"] > 0).astype(int)

inconsistent = df[
    df["Defective"] != expected_labels
]

print("Inconsistent bug/Defective mappings:", len(inconsistent))

print("\nSOURCE FILE LABEL CONSISTENCY")

file_labels = (
    df.groupby("File")["Defective"]
    .nunique(dropna=False)
)

conflicting_files = file_labels[file_labels > 1]

print("Files with conflicting labels:", len(conflicting_files))

if not conflicting_files.empty:
    print("\nConflicting files:")
    print(conflicting_files.to_string())

print("\nFINAL RESULT")

if (
    df["bug"].notna().all()
    and df["Defective"].notna().all()
    and invalid_labels.empty
    and inconsistent.empty
    and conflicting_files.empty
):
    print("PASS: Defect labels are internally consistent.")
else:
    print("WARNING: Label inconsistencies require investigation.")

print("\nOriginal datasets were not modified.")