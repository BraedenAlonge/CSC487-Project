# PokéGAN: Generating Original Pokémon Using a GAN

**CSC 487 Deep Learning - Final Project**  
**Stage 2: Baseline Model Implementation**

This repository contains a complete PyTorch implementation of a DCGAN (Deep Convolutional Generative Adversarial Network) for generating Pokémon-style sprite images.

## Project Overview

Our project aims to generate new sprite images of Pokémon-style creatures using a Generative Adversarial Network (GAN) trained from scratch. The model learns the different attributes that give Pokémon their unique style, including color schemes, outlines, shading, patterns, and body structure.

## Repository Structure

```
CSC487-Project/
├── models/                 # Model definitions
│   ├── __init__.py
│   ├── generator.py       # Generator network (DCGAN-style)
│   └── discriminator.py   # Discriminator network
├── data/                   # Data loading
│   ├── __init__.py
│   └── pokemon_dataset.py # Dataset class for Pokémon images
├── utils/                 # Utility functions
│   ├── __init__.py
│   ├── metrics.py         # FID, IS, Diversity Score
│   ├── visualization.py   # Plotting and visualization
│   └── reproducibility.py # Seed setting and determinism
├── configs/                # Configuration files
│   └── baseline.yaml      # Baseline model configuration
├── train.py               # Training script
├── eval.py                # Evaluation script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Setup Instructions

### For Local Development

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### For Google Colab (GPU Training)

See **[COLAB_SETUP.md](COLAB_SETUP.md)** for detailed instructions on running training on Google Colab with GPU support.

Quick start:
1. Upload your code to Colab (or clone from GitHub)
2. Install dependencies: `!pip install torch torchvision torchmetrics pyyaml matplotlib scipy tensorboard`
3. Enable GPU: Runtime → Change runtime type → GPU
4. Download dataset and update config paths
5. Run: `!python train.py --config configs/colab.yaml`

Or use the provided **[colab_notebook.ipynb](colab_notebook.ipynb)** notebook.

### 2. Prepare Dataset

1. Download the "1000 Pokémon Dataset" from Kaggle: https://www.kaggle.com/datasets/noodulz/pokemon-dataset-1000/data
2. Extract the dataset. The dataset comes with pre-split folders:
   - Extract to `data/pokemon/` (or your preferred location)
   - The dataset should contain `train/`, `val/`, and `test/` folders
3. Update the paths in `configs/baseline.yaml` if you extracted to a different location

The dataset structure should look like:
```
data/pokemon/
├── train/          # Training images
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── val/            # Validation images
│   ├── image1.png
│   └── ...
├── test/           # Test images (for final evaluation)
│   ├── image1.png
│   └── ...
├── dataset/        # Original dataset (if included)
├── generate_splits # Python script (optional, splits already done)
└── metadata.csv    # Dataset metadata (optional)
```

**Note:** The dataset already has train/val/test splits, so you don't need to run `generate_splits` unless you want to recreate the splits.

### 3. Verify Installation

Test the models:
```bash
python models/generator.py
python models/discriminator.py
```

## Usage

### Training

Train the baseline model:

```bash
python train.py --config configs/baseline.yaml
```

**Reproducibility requirement:**
```bash
python train.py --config configs/baseline.yaml
```

This will:
- Load and preprocess the dataset
- Train the generator and discriminator
- Save checkpoints every 25 epochs
- Generate sample images every 5 epochs
- Log metrics to TensorBoard
- Save training curves

**Training outputs:**
- `checkpoints/`: Model checkpoints (best_model.pt, final_model.pt, checkpoint_epoch_*.pt)
- `outputs/`: Generated sample images and training curves
- `logs/`: TensorBoard logs

### Evaluation

Evaluate a trained model:

```bash
python eval.py --checkpoint checkpoints/baseline.pt
```

Or with custom options:

```bash
python eval.py --checkpoint checkpoints/best_model.pt --n_samples 1000 --output_dir eval_outputs
```

**Evaluation outputs:**
- `eval_outputs/real_samples.png`: Grid of real images
- `eval_outputs/fake_samples.png`: Grid of generated images
- `eval_outputs/confusion_matrix.png`: Discriminator confusion matrix
- `eval_outputs/metrics.txt`: Quantitative metrics (FID, IS, Diversity)

### Resume Training

Resume training from a checkpoint:

```bash
python train.py --config configs/baseline.yaml --resume checkpoints/checkpoint_epoch_50.pt
```

## Model Architecture

### Generator
- **Input**: 100-dimensional Gaussian noise vector
- **Architecture**: 
  - Linear: 100 → 4×4×512
  - 4 TransposeConv2d blocks (4×4 → 8×8 → 16×16 → 32×32 → 64×64)
  - BatchNorm and ReLU activations
  - Tanh output activation (values in [-1, 1])
- **Output**: 64×64×3 RGB image
- **Parameters**: ~2.5-3 million

### Discriminator
- **Input**: 64×64×3 RGB image
- **Architecture**:
  - 4 Conv2d blocks with strided convolutions (64×64 → 32×32 → 16×16 → 8×8 → 4×4)
  - LeakyReLU activations (slope=0.2)
  - BatchNorm layers
  - Flatten → Linear → Sigmoid
- **Output**: Probability score (real vs. fake)
- **Parameters**: ~2.5-3 million

## Evaluation Metrics

1. **Fréchet Inception Distance (FID)**: Measures statistical similarity between real and generated images (lower is better)
2. **Inception Score (IS)**: Measures quality and diversity of generated images (higher is better)
3. **Diversity Score**: Measures variety of generated images using pairwise distances (higher is better)

## Configuration

Edit `configs/baseline.yaml` to customize:
- Model architecture (nz, ngf, ndf, nc)
- Training hyperparameters (batch_size, epochs, learning rates)
- Data paths and augmentation
- Logging and checkpoint intervals

## Reproducibility

The code includes:
- Fixed random seeds (configurable in config file)
- Deterministic operations option
- Checkpoint saving for exact model state
- Complete configuration files

To ensure reproducibility:
1. Use the same random seed (default: 42)
2. Use deterministic operations (set `deterministic: true` in config)
3. Use the same dataset and preprocessing

## Stage 2 Deliverables Checklist

- [x] Complete PyTorch training pipeline
- [x] Data loading and preprocessing
- [x] Model definition (no pretrained weights)
- [x] Training loop with logging, validation, and checkpointing
- [x] Evaluation on held-out data
- [x] Quantitative baseline results (FID, IS, Diversity)
- [x] Error analysis tools (confusion matrix, visualizations)
- [x] Training/validation curves
- [x] Reproducible codebase (train.py, eval.py, configs/)
- [x] Dependencies listed (requirements.txt)
- [x] One-click reproducibility

## Authors

- Lucas Summers (lsumme01@calpoly.edu)
- Braeden Alonge (balonge@calpoly.edu)

## License

This project is for educational purposes only. Pokémon sprites are copyrighted by Nintendo/Game Freak. Our use is educational/non-commercial under fair use doctrine.

