from pathlib import Path

import pandas as pd


def main():
    metadata_path = Path("data/processed/metadata.csv")

    if not metadata_path.exists():
        raise FileNotFoundError("Run src/inspect_dataset.py first.")

    df = pd.read_csv(metadata_path)

    print("Image validation report")
    print("-----------------------")

    total_samples = len(df)
    missing_width = df["image_width"].isna().sum()
    missing_height = df["image_height"].isna().sum()

    unique_sizes = df[["image_width", "image_height"]].drop_duplicates()

    print(f"Total samples: {total_samples}")
    print(f"Missing image widths: {missing_width}")
    print(f"Missing image heights: {missing_height}")

    print("\nUnique image sizes:")
    print(unique_sizes)

    expected_width = 28
    expected_height = 28

    invalid_sizes = df[
        (df["image_width"] != expected_width)
        | (df["image_height"] != expected_height)
    ]

    print(f"\nImages with unexpected size: {len(invalid_sizes)}")

    if len(invalid_sizes) == 0 and missing_width == 0 and missing_height == 0:
        print("\nImage validation passed.")
    else:
        print("\nImage validation issues found.")


if __name__ == "__main__":
    main()