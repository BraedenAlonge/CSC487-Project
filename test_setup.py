"""
Quick setup test script to verify installation and model architecture.
Run this before training to ensure everything is set up correctly.
"""

import torch
import sys
import os

print("="*50)
print("Testing PokéGAN Setup")
print("="*50)

# Test 1: PyTorch installation
print("\n1. Testing PyTorch installation...")
try:
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
    print("   ✓ PyTorch OK")
except Exception as e:
    print(f"   ✗ PyTorch error: {e}")
    sys.exit(1)

# Test 2: Model imports
print("\n2. Testing model imports...")
try:
    from models import Generator, Discriminator
    print("   ✓ Models imported successfully")
except Exception as e:
    print(f"   ✗ Model import error: {e}")
    sys.exit(1)

# Test 3: Generator architecture
print("\n3. Testing Generator architecture...")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    netG = Generator(nz=100, ngf=512, nc=3).to(device)
    noise = torch.randn(4, 100, device=device)
    with torch.no_grad():
        fake = netG(noise)
    assert fake.shape == (4, 3, 64, 64), f"Expected (4, 3, 64, 64), got {fake.shape}"
    assert -1.0 <= fake.min().item() <= 1.0, "Output should be in [-1, 1]"
    assert -1.0 <= fake.max().item() <= 1.0, "Output should be in [-1, 1]"
    print(f"   Generator output shape: {fake.shape}")
    print(f"   Generator parameters: {sum(p.numel() for p in netG.parameters()):,}")
    print("   ✓ Generator OK")
except Exception as e:
    print(f"   ✗ Generator error: {e}")
    sys.exit(1)

# Test 4: Discriminator architecture
print("\n4. Testing Discriminator architecture...")
try:
    netD = Discriminator(nc=3, ndf=64).to(device)
    real = torch.randn(4, 3, 64, 64, device=device)
    with torch.no_grad():
        output = netD(real)
    assert output.shape == (4, 1), f"Expected (4, 1), got {output.shape}"
    assert 0.0 <= output.min().item() <= 1.0, "Output should be in [0, 1]"
    assert 0.0 <= output.max().item() <= 1.0, "Output should be in [0, 1]"
    print(f"   Discriminator output shape: {output.shape}")
    print(f"   Discriminator parameters: {sum(p.numel() for p in netD.parameters()):,}")
    print("   ✓ Discriminator OK")
except Exception as e:
    print(f"   ✗ Discriminator error: {e}")
    sys.exit(1)

# Test 5: Dataset import
print("\n5. Testing dataset import...")
try:
    # Try different import methods for Colab compatibility
    try:
        from data import PokemonDataset
    except ImportError:
        # If that fails, try importing directly
        import sys
        import os
        # Add current directory to path if not already there
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        from data.pokemon_dataset import PokemonDataset
    print("   ✓ Dataset class imported successfully")
except Exception as e:
    print(f"   ✗ Dataset import error: {e}")
    print("   Note: This might be a path issue. Try running from the project root directory.")
    # Don't exit - this is a warning, not critical
    print("   (Continuing anyway - import will be checked during training)")

# Test 6: Utility imports
print("\n6. Testing utility imports...")
try:
    from utils.reproducibility import set_seed
    from utils.metrics import calculate_fid
    from utils.visualization import save_image_grid
    print("   ✓ Utilities imported successfully")
except Exception as e:
    print(f"   ✗ Utility import error: {e}")
    sys.exit(1)

# Test 7: Configuration file
print("\n7. Testing configuration file...")
try:
    import yaml
    # Check for colab.yaml first (if running in Colab), then baseline.yaml
    config_path = 'configs/colab.yaml' if os.path.exists('configs/colab.yaml') else 'configs/baseline.yaml'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"   ✓ Configuration file loaded: {config_path}")
        print(f"   - Model nz: {config['model']['nz']}")
        print(f"   - Batch size: {config['training']['batch_size']}")
        print(f"   - Epochs: {config['training']['epochs']}")
    else:
        print(f"   ⚠ Configuration file not found: {config_path}")
        print("   (This is OK if you haven't created it yet)")
except Exception as e:
    print(f"   ✗ Configuration error: {e}")

# Test 8: Data directory
print("\n8. Checking data directory...")
try:
    # Check for colab.yaml first (if running in Colab), then baseline.yaml
    config_path = 'configs/colab.yaml' if os.path.exists('configs/colab.yaml') else 'configs/baseline.yaml'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        data_dir = config['data'].get('train_dir', '')
        if data_dir and os.path.exists(data_dir):
            # Count images
            image_count = 0
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_count += 1
            print(f"   ✓ Data directory found: {data_dir}")
            print(f"   - Found {image_count} images")
        elif data_dir:
            print(f"   ⚠ Data directory not found: {data_dir}")
            print("   This is OK if you haven't downloaded the dataset yet.")
            print("   The dataset will be downloaded in Step 5 of the Colab notebook.")
        else:
            print("   ⚠ No train_dir specified in config")
    else:
        print("   ⚠ Cannot check data directory (config file not found)")
except Exception as e:
    print(f"   ⚠ Data directory check error: {e}")
    print("   (This is OK - dataset check is optional)")

print("\n" + "="*50)
print("Setup test complete!")
print("="*50)
print("\nIf all tests passed, you're ready to train:")
print("  python train.py --config configs/baseline.yaml")
print()

