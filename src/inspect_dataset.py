from pathlib import Path
import pandas as pd
import medmnist
from medmnist import INFO


def main():
    dataset_name = "pathmnist"
    data_dir = Path("data/raw")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info["python_class"])

    rows = []

    for split in ["train", "val", "test"]:
        dataset = DataClass(
            split=split,
            root=str(data_dir),
            download=False
        )

        for index in range(len(dataset)):
            image, label = dataset[index]

            rows.append({
                "dataset": dataset_name,
                "split": split,
                "sample_id": index,
                "image_width": image.size[0],
                "image_height": image.size[1],
                "label": int(label[0]),
                "label_name": info["label"][str(int(label[0]))]
            })

    metadata = pd.DataFrame(rows)
    metadata.to_csv(output_dir / "metadata.csv", index=False)

    print("Metadata saved to data/processed/metadata.csv")
    print(metadata.head())
    print(metadata["split"].value_counts())
    print(metadata["label_name"].value_counts())


if __name__ == "__main__":
    main()