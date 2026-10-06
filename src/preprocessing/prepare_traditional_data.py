import pandas as pd


AST_FILE = "data/raw/ast_features_labeled.csv"
PROMISE_FILE = "data/raw/ant-1.7.csv"
OUTPUT_FILE = "data/processed/traditional_metrics.csv"


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

ast_df = pd.read_csv(AST_FILE)
promise_df = pd.read_csv(PROMISE_FILE)

print("Datasets loaded successfully.")

print(f"AST dataset methods: {len(ast_df)}")
print(f"PROMISE classes:      {len(promise_df)}")


# --------------------------------------------------
# TRADITIONAL PROMISE METRICS
# --------------------------------------------------

TRADITIONAL_FEATURES = [
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


print("\nTraditional metrics:")
print(TRADITIONAL_FEATURES)


# --------------------------------------------------
# CREATE SIMPLE CLASS NAME
# --------------------------------------------------

# PROMISE uses fully-qualified class names such as:
#
# org.apache.tools.ant.taskdefs.Rmic
#
# Our AST dataset uses:
#
# Rmic
#
# Therefore, extract the final class name.

promise_df["Class"] = (
    promise_df["name"]
    .str.split(".")
    .str[-1]
)


# --------------------------------------------------
# CREATE SOURCE FILE NAME
# --------------------------------------------------

promise_df["File"] = (
    promise_df["Class"] + ".java"
)


# --------------------------------------------------
# CREATE BINARY DEFECT LABEL
# --------------------------------------------------

promise_df["Defective"] = (
    promise_df["bug"] > 0
).astype(int)


# --------------------------------------------------
# CHECK DUPLICATE FILE/CLASS COMBINATIONS
# --------------------------------------------------

duplicates = promise_df.duplicated(
    subset=["File", "Class"],
    keep=False
)

if duplicates.any():

    print("\nWARNING:")
    print("Duplicate File/Class combinations found:")

    print(
        promise_df.loc[
            duplicates,
            ["name", "File", "Class"]
        ].to_string(index=False)
    )

    raise ValueError(
        "Duplicate class mappings found. "
        "Traditional metrics cannot be safely "
        "merged until these are resolved."
    )


# --------------------------------------------------
# KEEP ONLY REQUIRED COLUMNS
# --------------------------------------------------

promise_metrics = promise_df[
    ["File", "Class"] +
    TRADITIONAL_FEATURES +
    ["bug", "Defective"]
]


# --------------------------------------------------
# MERGE WITH METHOD-LEVEL AST DATA
# --------------------------------------------------

traditional_df = ast_df[
    ["File", "Class", "Method"]
].merge(
    promise_metrics,
    on=["File", "Class"],
    how="left"
)


# --------------------------------------------------
# CHECK FOR UNMATCHED METHODS
# --------------------------------------------------

missing_metrics = traditional_df[
    TRADITIONAL_FEATURES
].isnull().any(axis=1)


print("\n----------------------------------------")
print("MERGE CHECK")
print("----------------------------------------")

print(
    f"Methods before merge: "
    f"{len(ast_df)}"
)

print(
    f"Methods after merge:  "
    f"{len(traditional_df)}"
)

print(
    f"Methods without traditional metrics: "
    f"{missing_metrics.sum()}"
)


if missing_metrics.any():

    print("\nUnmatched methods:")

    print(
        traditional_df.loc[
            missing_metrics,
            ["File", "Class", "Method"]
        ].head(20).to_string(index=False)
    )

    raise ValueError(
        "Some methods could not be matched "
        "with PROMISE traditional metrics."
    )


# --------------------------------------------------
# CHECK LABEL CONSISTENCY
# --------------------------------------------------

ast_labels = ast_df[
    ["File", "Class", "bug", "Defective"]
].drop_duplicates(
    subset=["File", "Class"]
)

label_check = traditional_df.merge(
    ast_labels,
    on=["File", "Class"],
    how="left",
    suffixes=("_traditional", "_ast")
)


bug_difference = (
    label_check["bug_traditional"]
    != label_check["bug_ast"]
)

defective_difference = (
    label_check["Defective_traditional"]
    != label_check["Defective_ast"]
)


print("\n----------------------------------------")
print("LABEL CONSISTENCY CHECK")
print("----------------------------------------")

print(
    f"Bug-label differences: "
    f"{bug_difference.sum()}"
)

print(
    f"Defective-label differences: "
    f"{defective_difference.sum()}"
)


if bug_difference.any() or defective_difference.any():

    raise ValueError(
        "Traditional metrics labels do not match "
        "the existing AST labels."
    )


# --------------------------------------------------
# KEEP FINAL DATASET
# --------------------------------------------------

traditional_df = traditional_df[
    ["File", "Class", "Method"] +
    TRADITIONAL_FEATURES +
    ["bug", "Defective"]
]


# --------------------------------------------------
# CREATE OUTPUT DIRECTORY
# --------------------------------------------------

import os

os.makedirs(
    "data/processed",
    exist_ok=True
)


# --------------------------------------------------
# SAVE DATASET
# --------------------------------------------------

traditional_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n========================================")
print("TRADITIONAL METRICS DATASET CREATED")
print("========================================")

print(
    f"Output file: {OUTPUT_FILE}"
)

print(
    f"Number of methods: "
    f"{len(traditional_df)}"
)

print(
    f"Number of files: "
    f"{traditional_df['File'].nunique()}"
)

print(
    f"Number of classes: "
    f"{traditional_df['Class'].nunique()}"
)

print("\nColumns:")
print(traditional_df.columns.tolist())

print("\nMissing values:")
print(traditional_df.isnull().sum())

print("\nDefective distribution:")
print(
    traditional_df["Defective"].value_counts()
)

print("\nDefective percentages:")
print(
    traditional_df["Defective"]
    .value_counts(normalize=True) * 100
)

print("\n========================================")
print("PREPROCESSING COMPLETE")
print("========================================")