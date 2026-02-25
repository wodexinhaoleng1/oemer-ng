# oemer-ng

A modern, fast Optical Music Recognition (OMR) package. This is a PyTorch port of the original [oemer](https://github.com/BreezeWhite/oemer) project.

## Features

- **Training Pipeline**: Complete training loop for semantic segmentation of music scores.
- **Dataset Support**: Built-in support for CvcMuscima-Distortions and DeepScoresV2 datasets.
- **Modern Architecture**: U-Net based model with focal tversky loss for handling class imbalance.
- **Inference**: Segmentation of music symbols and staff lines.

## Installation

```bash
# Clone the repository
git clone https://github.com/helium729/oemer-ng.git
cd oemer-ng

# Install in editable mode
pip install -e .
```

## Quick Start: Training with Sample Data

If you want to try the training pipeline without downloading gigabytes of data, we provide a script to generate sample data.

1.  **Generate Sample Data**:
    ```bash
    python scripts/create_sample_data.py
    ```
    This will create `data/sample_cvc` and `data/sample_ds2` directories.

2.  **Run Training**:
    ```bash
    # Train on CvcMuscima-style sample data
    python examples/train_model.py --dataset_path data/sample_cvc --dataset_type cvc --epochs 5

    # Train on DeepScores-style sample data
    python examples/train_model.py --dataset_path data/sample_ds2 --dataset_type ds2 --epochs 5
    ```

## Training on Real Datasets

To train a robust model, you should use the full datasets.

### 1. CVC-MUSCIMA (Staff Removal)

This dataset focuses on staff line removal and symbol segmentation on handwritten scores.

**Download:**
- Go to the [CVC-MUSCIMA Database page](http://pages.cvc.uab.es/cvcmuscima/index_database.html).
- Download the **Staff Removal set** (1.9 GB): [Direct Link](http://datasets.cvc.uab.es/muscima/CVCMUSCIMA_SR.zip).

**Preparation:**
1.  Extract the zip file. You should see folders like `CvcMuscima-Distortions` containing distortion types (e.g., `ideal`, `curvature`, `thickness-ratio`).
2.  Each distortion folder contains subfolders (`w-xx`) with `image`, `gt` (staff lines), and `symbol` (music symbols) directories.
3.  Point the training script to the root folder containing the distortion types.

**Training Command:**
```bash
python examples/train_model.py \
    --dataset_path /path/to/CvcMuscima-Distortions \
    --dataset_type cvc \
    --epochs 50 \
    --batch_size 8
```

### 2. DeepScoresV2 (Dense)

This dataset contains large-scale synthetic music scores with dense segmentation masks.

**Download:**
- Go to the [Zenodo Record](https://zenodo.org/records/4012193).
- Download `ds2_dense.tar.gz` (741 MB) for a smaller, manageable dataset.

**Preparation:**
1.  Extract the tarball.
2.  You should see an `images` directory and a `segmentation` directory (or similar structure).
3.  Ensure the directory structure matches:
    ```
    ds2_dense/
        images/
            *.png
        segmentation/
            *_seg.png
    ```

**Training Command:**
```bash
python examples/train_model.py \
    --dataset_path /path/to/ds2_dense \
    --dataset_type ds2 \
    --epochs 50 \
    --batch_size 4
```

## Inference

You can run inference using a trained model or the default initialized model.

```python
import torch
from oemer_ng import OMRPipeline

# Load your trained model
pipeline = OMRPipeline(model_path='checkpoints/final_model.pth', num_classes=3)

# Predict
image_path = 'path/to/sheet_music.png'
# Returns a dictionary with prediction map and confidence
result = pipeline.predict(image_path, return_probabilities=True)

prediction_map = result['prediction'] # (H, W) array of class indices
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License
