# PokéGAN: Generating Original Pokémon Sprites Using a GAN

**CSC 487 Deep Learning - Final Project (Stage 3)**

Our project aims to generate new sprite images of Pokémon-style creatures using a Deep Convolutional Generative Adversarial Network (DCGAN) trained from scratch. The model learns the different attributes that give Pokémon their unique style, including color schemes, outlines, shading, patterns, and body structure.

## Project Structure

```
CSC487-Project/
├── models/
│   ├── __init__.py
│   ├── generator.py        # Generator network (DCGAN-style)
│   ├── discriminator.py    # Discriminator network
│   └── attention.py        # Self-attention module (SAGAN-style)
├── data/                   # Data loading
│   ├── __init__.py
│   ├── pokemon_dataset.py  # Dataset class for Pokémon images
│   └── generate_splits.py  # Script to split dataset into train/val/test
├── utils/
│   ├── __init__.py
│   ├── metrics.py          # FID, IS, Diversity Score
│   ├── visualization.py    # Plotting and visualization
│   └── reproducibility.py  # Seed setting and determinism
├── configs/
│   ├── baseline.yaml       # Baseline model configuration
│   ├── final.yaml          # Final model configuration
├── train.py                # Training script
├── eval.py                 # Evaluation script
├── train_colab.ipynb       # Google Colab training notebook
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Setup Instructions (for local developement)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

1. Download the "1000 Pokémon Dataset" from Kaggle: https://www.kaggle.com/datasets/noodulz/pokemon-dataset-1000/data
2. Extract the dataset. The dataset comes with pre-split folders:
   - Extract to `data/` (or your preferred location)
   - The dataset should contain `train/`, `val/`, and `test/` folders
3. Update the paths in `configs/[config_file].yaml` if you extracted to a different location

The dataset structure should look like:
```
data/
├── train/             # Training images
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── val/               # Validation images
│   ├── image1.png
│   └── ...
├── test/              # Test images (for final evaluation)
│   ├── image1.png
│   └── ...
├── dataset/           # Original dataset (if included)
└── metadata.csv       # Dataset metadata (optional)
```

#### Notes:
- The dataset already has train/val/test splits, so you **don't need** to run `generate_splits` unless you want to recreate the splits.
- The `metadata.csv` file contains information about the Pokémon but isn't required for training
- The `dataset/` folder contains the original unsplit data (not needed if train/val/test exist)

### 3. Verify Installation

To test that the models and imports are correct:
```bash
python test_setup.py
```

## Setup Instructions (for Google Colab training)
Use the provided **[train_colab.ipynb](train_colab.ipynb)** notebook for GPU training on Google Colab. Just follow the instructions and run the cells!

## Usage
### Training

**Baseline model:**
```bash
python train.py --config configs/baseline.yaml
```

**Final model:**
```bash
python train.py --config configs/final.yaml
```

**Training outputs:**
- `checkpoints/`: Model checkpoints (best_model_epoch_X.pt, final_model_epoch_X.pt)
- `outputs/`: Generated sample images and training curves
- `logs/`: TensorBoard logs

### Resuming Training

Resume training from a checkpoint:
```bash
python train.py --config configs/baseline.yaml --resume checkpoints/best_model_epoch_50.pt
```

### Evaluation

Evaluate a trained model:
```bash
python eval.py --checkpoint checkpoints/best_model_epoch_100.pt --config configs/baseline.yaml
```

Or with custom options:
```bash
python eval.py --checkpoint checkpoints/best_model_epoch_100.pt --config configs/baseline.yaml --n_samples 1000 --output_dir eval_outputs
```

**Evaluation outputs:**
- `eval_outputs/real_eval_samples.png`: Grid of real images
- `eval_outputs/fake_eval_samples.png`: Grid of generated images
- `eval_outputs/confusion_matrix_eval.png`: Discriminator confusion matrix

## Model Architecture

### Generator
- **Input**: 100-dimensional Gaussian noise vector
- **Architecture**: 
  - Linear: 100 → 4×4×512
  - 4 TransposeConv2d blocks (4×4 → 8×8 → 16×16 → 32×32 → 64×64)
  - BatchNorm and ReLU activations
  - Optional self-attention at 16×16 or 32×32 spatial resolution
  - Optional dropout layers
  - Tanh output activation (values in [-1, 1])
- **Output**: 64×64×3 RGB image
- **Parameters**: ~2.5-3 million

### Discriminator
- **Input**: 64×64×3 RGB image
- **Architecture**:
  - 4 Conv2d blocks with strided convolutions (64×64 → 32×32 → 16×16 → 8×8 → 4×4)
  - LeakyReLU activations (slope=0.2)
  - BatchNorm layers (or Spectral Normalization if enabled)
  - Optional self-attention at 16×16 or 32×32 spatial resolution
  - Optional dropout layers
  - Flatten → Linear → Sigmoid
- **Output**: Probability score (real vs. fake)
- **Parameters**: ~2.5-3 million

### Advanced Features

**Self-Attention Mechanism:**
- SAGAN-style self-attention module for capturing long-range dependencies
- Can be enabled in both generator and discriminator
- Placed at configurable spatial resolutions (16×16 or 32×32)

**Spectral Normalization:**
- Optional spectral normalization for discriminator stability
- Helps stabilize GAN training by constraining discriminator weights

## Training Features

The training pipeline includes several advanced techniques for stable GAN training:

- **Label Smoothing**: Reduces discriminator overconfidence (configurable two-sided or one-sided)
- **Label Flipping**: Randomly flips labels with configurable probability to add noise
- **Gradient Clipping**: Prevents exploding gradients in generator and discriminator
- **Early Stopping**: Stops training when FID stops improving (configurable patience)
- **Data Augmentation**: Optional horizontal/vertical flips and rotation
- **Separate Learning Rates**: Different learning rates for generator and discriminator
- **Validation Monitoring**: FID calculated on validation set each epoch
- **Checkpointing**: Automatic saving of best and final models

## Evaluation Metrics

1. **Fréchet Inception Distance (FID)**: Measures statistical similarity between real and generated images (lower is better)
2. **Inception Score (IS)**: Measures quality and diversity of generated images (higher is better)
3. **Diversity Score**: Measures variety of generated images using pairwise L2 distances (higher is better)

## Configuration

The project includes three configuration files:

- **`configs/baseline.yaml`**: Baseline DCGAN configuration (Stage 2)
- **`configs/final.yaml`**: Final model with self-attention, spectral normalization, and advanced training features

Each config file allows customization of:
- Model architecture (nz, ngf, ndf, nc, attention, spectral norm, dropout)
- Training hyperparameters (batch_size, epochs, learning rates, beta1, beta2)
- Training techniques (label smoothing, gradient clipping, early stopping)
- Data paths and augmentation settings
- Logging and checkpoint intervals

## Reproducibility

The code includes comprehensive reproducibility features:
- Fixed random seeds (configurable in config file, default: 42)
- Optional deterministic operations (slower but fully reproducible)
- Checkpoint saving for exact model state
- Complete configuration files for all experiments

To ensure reproducibility:
1. Use the same random seed (default: 42)
2. Enable deterministic operations (set `deterministic: true` in config)
3. Use the same dataset and preprocessing
4. Use the same configuration file

## Authors

- Lucas Summers (lsumme01@calpoly.edu)
- Braeden Alonge (balonge@calpoly.edu)

## License

This project is for educational purposes only. Pokémon sprites are copyrighted by Nintendo/Game Freak. Our use is educational/non-commercial under fair use doctrine.
