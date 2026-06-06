from pathlib import Path
import pandas as pd


def main():
    metadata_path = Path("data/processed/metadata.csv")

    if not metadata_path.exists():
        raise FileNotFoundError("Run inspect_dataset.py first.")

    df = pd.read_csv(metadata_path)

    print("Label validation report")
    print("-----------------------")

    missing_labels = df["label"].isna().sum()
    print(f"Missing labels: {missing_labels}")

    print("\nClass distribution:")
    print(df["label_name"].value_counts())

    print("\nClass distribution by split:")
    print(pd.crosstab(df["label_name"], df["split"]))

    duplicated_ids = df.duplicated(subset=["split", "sample_id"]).sum()
    print(f"\nDuplicated sample IDs within splits: {duplicated_ids}")

    if missing_labels == 0 and duplicated_ids == 0:
        print("\nValidation passed.")
    else:
        print("\nValidation issues found.")


if __name__ == "__main__":
    main()