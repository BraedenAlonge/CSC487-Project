# Google Colab Setup Guide

This guide will help you run the PokéGAN training on Google Colab with GPU support.

## Step 1: Upload Your Code to Colab

### Option A: Using GitHub (Recommended)
1. Push your code to a GitHub repository
2. In Colab, clone the repository:
   ```python
   !git clone https://github.com/BraedenAlonge/CSC487-Project.git
   %cd CSC487-Project
   ```

### Option B: Direct Upload
1. Create a new Colab notebook
2. Upload your project files using the file browser (left sidebar)
3. Or zip your project and upload it, then extract:
   ```python
   from google.colab import files
   uploaded = files.upload()
   !unzip -q your-project.zip
   ```

## Step 2: Install Dependencies

Run this in a Colab cell:
```python
!pip install torch torchvision torchmetrics pyyaml matplotlib scipy tensorboard
```

## Step 3: Enable GPU

1. Go to **Runtime** → **Change runtime type**
2. Select **GPU** (T4 or A100 if available)
3. Click **Save**

Verify GPU is available:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

## Step 4: Upload Dataset

### Option A: Upload to Colab (Small datasets, temporary)
```python
from google.colab import files
# Upload your dataset zip file
uploaded = files.upload()
!unzip -q pokemon-dataset-1000.zip -d data/
```

### Option B: Use Google Drive (Recommended for large datasets)
1. Upload your dataset to Google Drive
2. Mount Google Drive in Colab:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Update config to point to Drive:
   ```python
   # In configs/baseline.yaml or create a new config
   train_dir: "/content/drive/MyDrive/pokemon-dataset-1000/train"
   val_dir: "/content/drive/MyDrive/pokemon-dataset-1000/val"
   test_dir: "/content/drive/MyDrive/pokemon-dataset-1000/test"
   ```

### Option C: Download from Kaggle (Recommended)
```python
# Install Kaggle API
!pip install kaggle

# Upload your kaggle.json file (get it from Kaggle Account → API)
from google.colab import files
files.upload()  # Upload kaggle.json

# Set up Kaggle credentials
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download dataset
!kaggle datasets download -d noodulz/pokemon-dataset-1000
!unzip -q pokemon-dataset-1000.zip -d data/
```

## Step 5: Update Configuration for Colab

Create a Colab-specific config or update paths:
```python
import yaml

# Read baseline config
with open('configs/baseline.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Update paths for Colab
config['data']['train_dir'] = '/content/CSC487-Project/data/pokemon-dataset-1000/train'
config['data']['val_dir'] = '/content/CSC487-Project/data/pokemon-dataset-1000/val'
config['data']['test_dir'] = '/content/CSC487-Project/data/pokemon-dataset-1000/test'

# Save Colab config
with open('configs/colab.yaml', 'w') as f:
    yaml.dump(config, f)
```

## Step 6: Run Training

```python
!python train.py --config configs/baseline.yaml
```

Or if using Colab config:
```python
!python train.py --config configs/colab.yaml
```

## Step 7: Save Checkpoints to Google Drive

To persist checkpoints across Colab sessions:

```python
# Mount drive
from google.colab import drive
drive.mount('/content/drive')

# Update config to save to Drive
config['training']['checkpoint_dir'] = '/content/drive/MyDrive/pokemon_gan/checkpoints'
config['training']['output_dir'] = '/content/drive/MyDrive/pokemon_gan/outputs'
config['training']['log_dir'] = '/content/drive/MyDrive/pokemon_gan/logs'
```

## Complete Colab Notebook Template

Here's a complete template you can use:

```python
# ============================================
# PokéGAN Training on Google Colab
# ============================================

# 1. Install dependencies
!pip install torch torchvision torchmetrics pyyaml matplotlib scipy tensorboard

# 2. Clone repository (or upload files)
!git clone https://github.com/yourusername/CSC487-Project.git
%cd CSC487-Project

# 3. Verify GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 4. Mount Google Drive (optional, for saving checkpoints)
from google.colab import drive
drive.mount('/content/drive')

# 5. Download dataset from Kaggle
!pip install kaggle
from google.colab import files
files.upload()  # Upload kaggle.json
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d noodulz/pokemon-dataset-1000
!unzip -q pokemon-dataset-1000.zip -d data/

# 6. Update config for Colab paths
import yaml
with open('configs/baseline.yaml', 'r') as f:
    config = yaml.safe_load(f)

config['data']['train_dir'] = '/content/CSC487-Project/data/pokemon-dataset-1000/train'
config['data']['val_dir'] = '/content/CSC487-Project/data/pokemon-dataset-1000/val'
config['data']['test_dir'] = '/content/CSC487-Project/data/pokemon-dataset-1000/test'

# Optional: Save checkpoints to Drive
config['training']['checkpoint_dir'] = '/content/drive/MyDrive/pokemon_gan/checkpoints'
config['training']['output_dir'] = '/content/drive/MyDrive/pokemon_gan/outputs'
config['training']['log_dir'] = '/content/drive/MyDrive/pokemon_gan/logs'

with open('configs/colab.yaml', 'w') as f:
    yaml.dump(config, f)

# 7. Run training
!python train.py --config configs/colab.yaml

# 8. View TensorBoard (optional)
# %load_ext tensorboard
# %tensorboard --logdir logs
```

## Tips for Colab

1. **Session Timeout**: Colab sessions disconnect after ~90 minutes of inactivity. Save checkpoints frequently!

2. **Storage Limits**: Free Colab has ~15GB storage. Use Google Drive for large datasets/checkpoints.

3. **GPU Limits**: Free tier gets T4 GPU (limited hours/day). Colab Pro gets A100 and more hours.

4. **Resume Training**: If session disconnects, resume from checkpoint:
   ```python
   !python train.py --config configs/colab.yaml --resume /content/drive/MyDrive/pokemon_gan/checkpoints/checkpoint_epoch_50.pt
   ```

5. **Monitor Training**: Use TensorBoard or print statements. Colab can display images inline.

6. **Download Results**: Download checkpoints/images when done:
   ```python
   from google.colab import files
   files.download('checkpoints/best_model.pt')
   ```

## Troubleshooting

- **CUDA out of memory**: Reduce `batch_size` in config (try 32 or 16)
- **Dataset not found**: Check paths are absolute (start with `/content/`)
- **Import errors**: Make sure you're in the project directory (`%cd CSC487-Project`)
- **Session disconnected**: Save checkpoints to Drive, resume training later

