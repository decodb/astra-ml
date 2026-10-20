
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data/raw/ast_features_with_signatures.csv"

OUTPUT = (
    ROOT / "data/processed/duplicate_diagnostics"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

print("=" * 65)
print("METHOD SIGNATURE DUPLICATE AUDIT")
print("=" * 65)

print("\nTotal methods:", len(df))

# Old method identity
old_key = ["File", "Class", "Method"]

# New method identity
new_key = [
    "File",
    "Package",
    "Class",
    "MethodSignature"
]

# Check old repeated method keys
old_duplicates = df.duplicated(
    subset=old_key,
    keep=False
)

print("\nOLD METHOD KEY")
print("Rows involved in repeated keys:", old_duplicates.sum())

# Check new repeated method keys
new_duplicates = df.duplicated(
    subset=new_key,
    keep=False
)

print("\nNEW METHOD KEY")
print("Rows involved in repeated keys:", new_duplicates.sum())

# Count repeated groups
repeated_groups = (
    df.groupby(new_key, dropna=False)
    .size()
    .reset_index(name="Occurrences")
)

repeated_groups = repeated_groups[
    repeated_groups["Occurrences"] > 1
]

print("Repeated signature groups:", len(repeated_groups))

# Save repeated signatures
repeated_groups.to_csv(
    OUTPUT / "repeated_signatures.csv",
    index=False
)

# Save all rows belonging to repeated signatures
repeated_rows = df.loc[new_duplicates].copy()

repeated_rows.to_csv(
    OUTPUT / "repeated_signature_rows.csv",
    index=False
)

# Check exact duplicate rows
exact_duplicates = df.duplicated(
    keep="first"
).sum()

print("\nEXACT DUPLICATES")
print("Excess exact duplicate rows:", exact_duplicates)

# Compare old and new identity counts
old_unique = df[old_key].drop_duplicates().shape[0]
new_unique = df[new_key].drop_duplicates().shape[0]

print("\nIDENTITY COMPARISON")
print("Unique old method keys:", old_unique)
print("Unique new method keys:", new_unique)
print("Additional methods distinguished:", new_unique - old_unique)

# Check whether repeated signatures have differing features
features = [
    "If", "For", "While", "Return",
    "TryCatch", "MethodCalls", "Variables",
    "Parameters", "ASTDepth",
    "BranchingFactor", "LOC"
]

conflicts = []

for key, group in df.groupby(new_key, dropna=False):

    if len(group) < 2:
        continue

    differing = [
        col for col in features
        if group[col].nunique(dropna=False) > 1
    ]

    conflicts.append({
        "File": key[0],
        "Package": key[1],
        "Class": key[2],
        "MethodSignature": key[3],
        "Occurrences": len(group),
        "DifferentFeatures": ", ".join(differing),
        "FeatureConflict": len(differing) > 0
    })

conflict_df = pd.DataFrame(conflicts)

conflict_df.to_csv(
    OUTPUT / "signature_conflicts.csv",
    index=False
)

print("\nSIGNATURE CONFLICTS")

if not conflict_df.empty:
    print(
        "Repeated signatures with differing AST features:",
        conflict_df["FeatureConflict"].sum()
    )

    print("\nExamples:")
    print(
        conflict_df[
            conflict_df["FeatureConflict"]
        ].head(10).to_string(index=False)
    )
else:
    print("No repeated signatures found.")

print("\nAudit complete.")
print("Original datasets were not modified.")
