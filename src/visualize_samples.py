from pathlib import Path

import matplotlib.pyplot as plt
import medmnist
from medmnist import INFO


def main():
    dataset_name = "pathmnist"
    data_dir = "data/raw"
    output_dir = Path("data/reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info["python_class"])

    dataset = DataClass(split="train", root=data_dir, download=False)

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))

    for i, ax in enumerate(axes.flat):
        image, label = dataset[i]
        label_id = int(label[0])
        label_name = info["label"][str(label_id)]

        ax.imshow(image)
        ax.set_title(label_name, fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    output_path = output_dir / "sample_visualization.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()