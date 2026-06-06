# MedImage DataOps

Medical imaging dataset preparation and validation pipeline for AI workflows.

## Overview

MedImage DataOps is a Python-based data management project focused on preparing high-quality medical imaging datasets for AI development workflows.

The project demonstrates dataset preparation, metadata generation, image validation, annotation/label quality checks, structured documentation, and reporting.

## Why This Project Matters

Medical AI systems depend on high-quality datasets. Before model development, datasets must be inspected, validated, documented, and prepared in a reproducible way.

This project simulates an AI data management workflow used in medical imaging R&D.

## Features

- Download public medical imaging dataset
- Generate structured metadata
- Validate image dimensions
- Validate labels and class distribution
- Detect duplicate sample IDs
- Generate dataset quality report
- Visualize medical image samples
- Document dataset assumptions and limitations

## Dataset

This project uses PathMNIST from MedMNIST.

PathMNIST is a histopathology image classification dataset with 9 tissue classes.

## Project Structure

```text
medimage-dataops/
├── data/
│   ├── raw/
│   ├── processed/
│   └── reports/
├── notebooks/
├── src/
│   ├── download_data.py
│   ├── inspect_dataset.py
│   ├── validate_images.py
│   ├── validate_labels.py
│   ├── visualize_samples.py
│   └── generate_report.py
├── dataset_card.md
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Pipeline

```bash
python src/download_data.py
python src/inspect_dataset.py
python src/validate_images.py
python src/validate_labels.py
python src/visualize_samples.py
python src/generate_report.py
```

## Outputs

The pipeline creates:

```text
data/processed/metadata.csv
data/reports/dataset_quality_report.md
data/reports/sample_visualization.png
```

## Baseline Model Results

A simple CNN baseline was trained on PathMNIST for 3 epochs.

| Metric | Result |
|---|---|
| Test accuracy | 74.83% |
| Macro F1-score | 0.69 |
| Weighted F1-score | 0.75 |

The baseline model confirms that the prepared dataset can be used successfully in an AI training workflow.

## Visual Outputs

### Sample Medical Images

![Sample Visualization](data/reports/sample_visualization.png)

### Training Loss

![Training Loss](data/reports/training_loss.png)

### Confusion Matrix

![Confusion Matrix](data/reports/confusion_matrix.png)

## Disclaimer

This project is for portfolio and educational purposes only. It is not intended for clinical decision-making.