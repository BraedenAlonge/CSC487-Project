# Dataset Setup Guide

## Quick Setup Steps

1. **Extract the dataset** to `data/pokemon/` (or your preferred location)

2. **Verify the structure** - You should see:
   ```
   data/pokemon/
   ├── train/       ← Training images (use this for training)
   ├── val/         ← Validation images (use this for validation)
   ├── test/        ← Test images (use this for final evaluation)
   ├── dataset/     ← Original dataset (optional)
   ├── generate_splits  ← Python script (optional, splits already done)
   └── metadata.csv ← Dataset metadata (optional)
   ```

3. **Update config if needed** - The default config (`configs/baseline.yaml`) expects:
   - `train_dir: "data/pokemon/train"`
   - `val_dir: "data/pokemon/val"`
   - `test_dir: "data/pokemon/test"`
   
   If you extracted to a different location, update these paths in the config file.

4. **You're ready to train!**
   ```bash
   python train.py --config configs/baseline.yaml
   ```

## What Each Folder Is For

- **`train/`**: Used during training to learn the generator and discriminator
- **`val/`**: Used during training to monitor performance and select best model
- **`test/`**: Used only during final evaluation (not used during training)

## Notes

- The dataset already has train/val/test splits, so you **don't need** to run `generate_splits`
- The `metadata.csv` file contains information about the Pokémon but isn't required for training
- The `dataset/` folder contains the original unsplit data (not needed if train/val/test exist)

