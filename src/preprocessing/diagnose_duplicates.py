
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "ast_features_labeled.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "duplicate_diagnostics"

KEY_COLUMNS = ["File", "Class", "Method"]

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

ALL_COLUMNS = [
    "File", "Package", "Class", "Method",
    *AST_FEATURES, "bug", "Defective"
]


def main():
    # This script only reads the input and writes separate reports.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DUPLICATE METHOD DIAGNOSTIC")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}\n"
            "Run this script from your astra-ml project."
        )

    df = pd.read_csv(INPUT_FILE)

    missing_columns = [
        column for column in ALL_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Expected columns are missing: {missing_columns}"
        )

    # Original CSV row number, including the header as row 1.
    # This is for tracing records back to the input file.
    df.insert(0, "SourceCSVRow", range(2, len(df) + 2))

    print(f"\nInput file: {INPUT_FILE}")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns) - 1}")

    # --------------------------------------------------
    # 2. Exact duplicate rows
    # --------------------------------------------------
    print("\n" + "=" * 70)
    print("CHECK A: EXACT DUPLICATE ROWS")
    print("=" * 70)

    # Exclude SourceCSVRow because it is diagnostic metadata,
    # not part of the original dataset.
    original_columns = [
        column for column in df.columns
        if column != "SourceCSVRow"
    ]

    exact_mask = df.duplicated(
        subset=original_columns,
        keep=False
    )

    exact_rows = df.loc[exact_mask].copy()

    # Number of rows that would remain after keeping one copy
    # of each distinct original record.
    duplicate_excess = df.duplicated(
        subset=original_columns,
        keep="first"
    ).sum()

    print(f"Rows involved in exact duplicates: {len(exact_rows)}")
    print(f"Excess repeated copies: {duplicate_excess}")
    print(
        "Distinct records represented in duplicate groups:",
        exact_rows[original_columns].drop_duplicates().shape[0]
    )

    exact_rows.to_csv(
        OUTPUT_DIR / "exact_duplicate_rows.csv",
        index=False
    )

    # --------------------------------------------------
    # 3. Repeated File/Class/Method keys
    # --------------------------------------------------
    print("\n" + "=" * 70)
    print("CHECK B: REPEATED METHOD KEYS")
    print("=" * 70)

    key_counts = (
        df.groupby(KEY_COLUMNS, dropna=False)
        .size()
        .reset_index(name="Occurrences")
    )

    repeated_keys = key_counts[
        key_counts["Occurrences"] > 1
    ].copy()

    repeated_keys = repeated_keys.sort_values(
        ["Occurrences", *KEY_COLUMNS],
        ascending=[False, True, True, True],
        na_position="last"
    )

    print(f"Distinct method keys: {len(key_counts)}")
    print(f"Repeated method-key groups: {len(repeated_keys)}")
    print(
        "Rows involved in repeated keys:",
        int(repeated_keys["Occurrences"].sum())
    )
    print(
        "Excess rows from repeated keys:",
        int((repeated_keys["Occurrences"] - 1).sum())
    )

    # Add a key-group identifier to make reports easier to inspect.
    repeated_keys.insert(
        0, "KeyGroupID",
        range(1, len(repeated_keys) + 1)
    )

    repeated_keys.to_csv(
        OUTPUT_DIR / "repeated_method_key_groups.csv",
        index=False
    )

    # Retrieve every row belonging to a repeated key.
    repeated_rows = df.merge(
        repeated_keys[KEY_COLUMNS + ["KeyGroupID", "Occurrences"]],
        on=KEY_COLUMNS,
        how="inner"
    )

    repeated_rows = repeated_rows.sort_values(
        ["KeyGroupID", "SourceCSVRow"]
    )

    repeated_rows.to_csv(
        OUTPUT_DIR / "repeated_method_key_rows.csv",
        index=False
    )

    # --------------------------------------------------
    # 4. Compare parameter counts and AST features
    # --------------------------------------------------
    print("\n" + "=" * 70)
    print("CHECK C: FEATURE DIFFERENCES WITHIN REPEATED KEYS")
    print("=" * 70)

    comparison_records = []

    for key, group in df.groupby(KEY_COLUMNS, dropna=False):
        if len(group) < 2:
            continue

        # Include missing values as a distinct value when
        # determining whether a feature differs.
        different_features = [
            feature
            for feature in AST_FEATURES
            if group[feature].nunique(dropna=False) > 1
        ]

        parameter_values = sorted(
            group["Parameters"].dropna().unique().tolist()
        )

        parameter_counts_differ = (
            group["Parameters"].nunique(dropna=False) > 1
        )

        # A useful warning sign: identical key, but different
        # AST features. It is not proof of an extraction error.
        comparison_records.append({
            "File": key[0],
            "Class": key[1],
            "Method": key[2],
            "Occurrences": len(group),
            "SourceCSVRows": ", ".join(
                map(str, group["SourceCSVRow"].tolist())
            ),
            "DifferentParameters": parameter_counts_differ,
            "ParameterValues": ", ".join(
                map(str, parameter_values)
            ),
            "DifferentASTFeatures": ", ".join(different_features),
            "NumberOfDifferentASTFeatures": len(different_features),
            "DifferentBugCounts": (
                group["bug"].nunique(dropna=False) > 1
            ),
            "DifferentDefectiveLabels": (
                group["Defective"].nunique(dropna=False) > 1
            ),
            "ExactDuplicateRowsInGroup": (
                group.duplicated(
                    subset=original_columns,
                    keep=False
                ).sum()
            ),
        })

    comparisons = pd.DataFrame(comparison_records)

    if not comparisons.empty:
        comparisons = comparisons.sort_values(
            [
                "NumberOfDifferentASTFeatures",
                "Occurrences"
            ],
            ascending=[False, False]
        )

        print(
            "Repeated-key groups with different parameter counts:",
            int(comparisons["DifferentParameters"].sum())
        )
        print(
            "Repeated-key groups with different AST features:",
            int(
                (
                    comparisons["NumberOfDifferentASTFeatures"] > 0
                ).sum()
            )
        )
        print(
            "Repeated-key groups with different bug counts:",
            int(comparisons["DifferentBugCounts"].sum())
        )

        print("\nTop 15 repeated keys with feature differences:")

        display_columns = [
            "File", "Class", "Method", "Occurrences",
            "ParameterValues", "DifferentASTFeatures",
            "SourceCSVRows"
        ]

        interesting = comparisons[
            comparisons["NumberOfDifferentASTFeatures"] > 0
        ]

        if interesting.empty:
            print("No repeated keys have differing AST features.")
        else:
            print(
                interesting[display_columns]
                .head(15)
                .to_string(index=False)
            )
    else:
        print("No repeated method keys found.")

    comparisons.to_csv(
        OUTPUT_DIR / "repeated_key_feature_comparison.csv",
        index=False
    )

    # --------------------------------------------------
    # 5. Check repeated names across different parameter counts
    # --------------------------------------------------
    print("\n" + "=" * 70)
    print("CHECK D: POSSIBLE OVERLOADED METHODS")
    print("=" * 70)

    # Group by file and class, but not method parameters.
    # If the same method name occurs multiple times and the
    # Parameters feature differs, it may be an overload.
    overload_records = []

    for key, group in df.groupby(
        ["File", "Class", "Method"],
        dropna=False
    ):
        if len(group) < 2:
            continue

        distinct_parameters = group["Parameters"].dropna().unique()

        if len(distinct_parameters) > 1:
            overload_records.append({
                "File": key[0],
                "Class": key[1],
                "Method": key[2],
                "Occurrences": len(group),
                "ParameterCounts": ", ".join(
                    map(
                        str,
                        sorted(distinct_parameters.tolist())
                    )
                ),
                "SourceCSVRows": ", ".join(
                    map(str, group["SourceCSVRow"].tolist())
                ),
                "Warning": (
                    "Possible overload, but inspect source code "
                    "to confirm."
                )
            })

    overloads = pd.DataFrame(overload_records)

    print(f"Possible overload groups: {len(overloads)}")

    if not overloads.empty:
        print(overloads.head(20).to_string(index=False))

    overloads.to_csv(
        OUTPUT_DIR / "possible_overloaded_methods.csv",
        index=False
    )

    # --------------------------------------------------
    # 6. Summary
    # --------------------------------------------------
    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    print(f"Reports saved in: {OUTPUT_DIR}")
    print("No source dataset rows were deleted or modified.")

    print("\nReports created:")
    for report in sorted(OUTPUT_DIR.glob("*.csv")):
        print(f" - {report.name}")


if __name__ == "__main__":
    main()
