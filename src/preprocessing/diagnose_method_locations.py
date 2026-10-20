
import pandas as pd
from pathlib import Path

# Load diagnostic dataset
df = pd.read_csv(
    "data/raw/ast_features_with_line_numbers.csv"
)

# Method identity without source location
method_key = [
    "File",
    "Package",
    "DeclaringType",
    "MethodSignature"
]

# Existing AST features
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

print("=" * 65)
print("METHOD SOURCE LOCATION AUDIT")
print("=" * 65)

print(f"\nTotal methods: {len(df)}")

# Find methods with repeated identities
repeated = df[
    df.duplicated(subset=method_key, keep=False)
].copy()

print(
    "Rows involved in repeated identities:",
    len(repeated)
)

# Identify conflicts
conflicting_rows = []
conflicting_groups = 0

for key, group in repeated.groupby(
    method_key, dropna=False
):
    different_features = [
        feature
        for feature in ast_features
        if group[feature].nunique(dropna=False) > 1
    ]

    if different_features:
        conflicting_groups += 1
        conflicting_rows.append(group)

print(
    "Repeated identities with conflicting AST features:",
    conflicting_groups
)

# Combine all conflicting rows
if conflicting_rows:
    conflicts = pd.concat(
        conflicting_rows,
        ignore_index=True
    )
else:
    conflicts = df.iloc[0:0].copy()

# Check whether line numbers distinguish these methods
location_key = method_key + [
    "StartLine",
    "EndLine"
]

print("\nSOURCE LOCATION COMPARISON")
print(
    "Unique identities without locations:",
    df[method_key].drop_duplicates().shape[0]
)

print(
    "Unique identities with locations:",
    df[location_key].drop_duplicates().shape[0]
)

print(
    "Repeated rows after adding locations:",
    df.duplicated(
        subset=location_key,
        keep=False
    ).sum()
)

print(
    "Excess rows with identical identity and location:",
    df.duplicated(
        subset=location_key
    ).sum()
)

# Display conflicting methods and their source locations
columns = [
    "File",
    "DeclaringType",
    "MethodSignature",
    "StartLine",
    "EndLine"
]

print("\nCONFLICTING METHOD LOCATIONS")

if not conflicts.empty:
    print(
        conflicts[columns]
        .sort_values(
            ["File", "MethodSignature", "StartLine"]
        )
        .to_string(index=False)
    )
else:
    print("No conflicting methods found.")

# Save diagnostic report
output_dir = Path(
    "data/processed/duplicate_diagnostics"
)
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "method_location_conflicts.csv"

conflicts.to_csv(
    output_path,
    index=False
)

print("\nReport saved to:", output_path)
print("Original datasets were not modified.")
