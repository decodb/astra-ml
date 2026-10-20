
import pandas as pd

# Load the new diagnostic dataset
df = pd.read_csv(
    "data/raw/ast_features_with_declaring_types.csv"
)

print("=" * 65)
print("DECLARING TYPE DUPLICATE AUDIT")
print("=" * 65)

print(f"\nTotal methods: {len(df)}")

# Previous identity
old_key = [
    "File",
    "Package",
    "Class",
    "MethodSignature"
]

# New identity
new_key = [
    "File",
    "Package",
    "DeclaringType",
    "MethodSignature"
]

# Compare repeated identities
old_duplicates = df.duplicated(
    subset=old_key, keep=False
)

new_duplicates = df.duplicated(
    subset=new_key, keep=False
)

print("\nOLD SIGNATURE KEY")
print(
    "Rows involved in repeated keys:",
    old_duplicates.sum()
)

print("\nNEW DECLARING-TYPE KEY")
print(
    "Rows involved in repeated keys:",
    new_duplicates.sum()
)

# Count repeated groups
groups = df.groupby(
    new_key, dropna=False
).size()

repeated_groups = groups[groups > 1]

print(
    "Repeated identity groups:",
    len(repeated_groups)
)

# Compare unique identities
old_unique = df[old_key].drop_duplicates().shape[0]
new_unique = df[new_key].drop_duplicates().shape[0]

print("\nIDENTITY COMPARISON")
print("Unique old identities:", old_unique)
print("Unique new identities:", new_unique)
print(
    "Additional identities distinguished:",
    new_unique - old_unique
)

# Exact duplicates
print("\nEXACT DUPLICATES")
print(
    "Excess exact duplicate rows:",
    df.duplicated().sum()
)

# Investigate conflicting AST features
ast_features = [
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

conflicts = []

for key, group in df.groupby(
    new_key, dropna=False
):

    if len(group) <= 1:
        continue

    different_features = [
        feature
        for feature in ast_features
        if group[feature].nunique(dropna=False) > 1
    ]

    if different_features:
        conflicts.append({
            "File": key[0],
            "Package": key[1],
            "DeclaringType": key[2],
            "MethodSignature": key[3],
            "Occurrences": len(group),
            "DifferentFeatures": ", ".join(
                different_features
            )
        })

conflicts_df = pd.DataFrame(conflicts)

print("\nDECLARING-TYPE CONFLICTS")
print(
    "Repeated identities with differing AST features:",
    len(conflicts_df)
)

if not conflicts_df.empty:
    print("\nExamples:")
    print(
        conflicts_df.head(10).to_string(index=False)
    )

print("\nAudit complete.")
print("Original datasets were not modified.")
