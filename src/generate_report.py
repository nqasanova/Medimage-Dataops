from pathlib import Path
import pandas as pd


def main():
    metadata_path = Path("data/processed/metadata.csv")
    report_path = Path("data/reports/dataset_quality_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if not metadata_path.exists():
        raise FileNotFoundError("Run inspect_dataset.py first.")

    df = pd.read_csv(metadata_path)

    total_samples = len(df)
    split_counts = df["split"].value_counts()
    label_counts = df["label_name"].value_counts()
    missing_labels = df["label"].isna().sum()
    duplicate_count = df.duplicated(subset=["split", "sample_id"]).sum()

    report = f"""# Dataset Quality Report

## Project
MedImage DataOps

## Dataset
PathMNIST from MedMNIST.

## Purpose
This report documents dataset quality checks for a medical imaging AI workflow.

## Total Samples
{total_samples}

## Split Distribution
{split_counts.to_markdown()}

## Label Distribution
{label_counts.to_markdown()}

## Validation Checks

| Check | Result |
|---|---|
| Missing labels | {missing_labels} |
| Duplicate sample IDs | {duplicate_count} |
| Metadata generated | Yes |
| Train/validation/test split available | Yes |

## Notes
This dataset is used for portfolio and educational purposes only.  
It is not intended for clinical decision-making.
"""

    report_path.write_text(report, encoding="utf-8")

    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()