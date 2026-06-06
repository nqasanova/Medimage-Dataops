from pathlib import Path
import medmnist
from medmnist import INFO


def main():
    dataset_name = "pathmnist"
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info["python_class"])

    for split in ["train", "val", "test"]:
        dataset = DataClass(
            split=split,
            root=str(data_dir),
            download=True
        )

        print(f"{split}: {len(dataset)} samples")

    print("Dataset downloaded successfully.")
    print(f"Dataset: {dataset_name}")
    print(f"Task: {info['task']}")
    print(f"Number of classes: {len(info['label'])}")


if __name__ == "__main__":
    main()