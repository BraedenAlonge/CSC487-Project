import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
from models import Generator, Discriminator
from data import PokemonDataset
from utils.reproducibility import set_seed
from utils.visualization import save_image_grid, plot_confusion_matrix
from utils.metrics import calculate_fid, calculate_inception_score, calculate_diversity_score

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate DCGAN for Pokémon generation')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/baseline.yaml',
                       help='Path to configuration file')
    parser.add_argument('--test_dir', type=str, default=None,
                       help='Path to test data directory (overrides config)')
    parser.add_argument('--n_samples', type=int, default=1000,
                       help='Number of samples to generate for evaluation')
    parser.add_argument('--output_dir', type=str, default='eval_outputs',
                       help='Directory to save evaluation outputs')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model(checkpoint_path, config, device):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create generator and discriminator models
    netG = Generator(
        nz=config['model']['nz'],
        ngf=config['model']['ngf'],
        nc=config['model']['nc'],
        kernel_size=config['model'].get('kernel_size', 4),
        stride=config['model'].get('stride', 2),
        padding=config['model'].get('padding', 1),
        dropout=config['model'].get('dropout_g', 0.0),
        attention=config['model'].get('attention_g', False),
        attention_layer=config['model'].get('attention_g_layer', 32)
    ).to(device)
    
    netD = Discriminator(
        nc=config['model']['nc'],
        ndf=config['model']['ndf'],
        kernel_size=config['model'].get('kernel_size', 4),
        stride=config['model'].get('stride', 2),
        padding=config['model'].get('padding', 1),
        dropout=config['model'].get('dropout_d', 0.0),
        use_spectral_norm=config['model'].get('use_spectral_norm', False),
        attention=config['model'].get('attention_d', False),
        attention_layer=config['model'].get('attention_d_layer', 32)
    ).to(device)
    
    # Load weights
    netG.load_state_dict(checkpoint['netG_state_dict'])
    netD.load_state_dict(checkpoint['netD_state_dict'])
    
    netG.eval()
    netD.eval()
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    return netG, netD

def generate_samples(netG, n_samples, nz, device, batch_size=64):
    print(f"Generating {n_samples} samples...")
    fake_images = []
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_size_actual = min(batch_size, n_samples - i)
            noise = torch.randn(batch_size_actual, nz, device=device)
            fake_batch = netG(noise)
            fake_images.append(fake_batch.cpu())
    
    fake_images = torch.cat(fake_images, dim=0)
    return fake_images

def evaluate_discriminator(netD, real_images, fake_images, device):
    real_scores = []
    fake_scores = []
    
    batch_size = 64
    with torch.no_grad():
        # Real images
        for i in range(0, len(real_images), batch_size):
            batch = real_images[i:i+batch_size].to(device)
            scores = netD(batch)
            real_scores.append(scores.cpu())
        
        # Fake images
        for i in range(0, len(fake_images), batch_size):
            batch = fake_images[i:i+batch_size].to(device)
            scores = netD(batch)
            fake_scores.append(scores.cpu())
    
    real_scores = torch.cat(real_scores, dim=0)
    fake_scores = torch.cat(fake_scores, dim=0)
    
    return real_scores, fake_scores

def main():
    # Prepare environment for evaluation
    args = parse_args()
    config = load_config(args.config)
    set_seed(config['training']['seed'])    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and test data
    netG, netD = load_model(args.checkpoint, config, device)    
    test_dir = args.test_dir if args.test_dir else config['data'].get('test_dir', config['data']['train_dir'])
    test_dataset = PokemonDataset(root_dir=test_dir, augment=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    print(f"Test dataset size: {len(test_dataset)}")
    print("Collecting real images...")
    real_images = []
    for batch in test_loader:
        real_images.append(batch)
    real_images = torch.cat(real_images, dim=0)
    
    # Limit real images for comparison to the number of generated samples
    n_eval = min(args.n_samples, len(real_images))
    real_images = real_images[:n_eval]
    print(f"Using {n_eval} real images for evaluation")
    
    # Generate fake images and save sample grids
    fake_images = generate_samples(netG, args.n_samples, config['model']['nz'], device)    
    print("Saving sample images...")
    save_image_grid(real_images[:64], os.path.join(args.output_dir, 'real_eval_samples.png'))
    save_image_grid(fake_images[:64], os.path.join(args.output_dir, 'fake_eval_samples.png'))

    # Calculate metrics
    print("Calculating metrics...")
    fid = calculate_fid(real_images.to(device), fake_images.to(device), device=device)    
    is_mean, is_std = calculate_inception_score(fake_images.to(device), device=device)
    diversity = calculate_diversity_score(fake_images.to(device), device=device)
    print("Metrics:")
    print(f"  FID: {fid:.4f} (lower is better)")
    print(f"  Inception Score: {is_mean:.4f} ± {is_std:.4f} (higher is better)")
    print(f"  Diversity Score: {diversity:.4f} (higher is better)")

    # Discriminator evaluation
    print("Evaluating discriminator...")
    real_scores, fake_scores = evaluate_discriminator(netD, real_images, fake_images, device)
    print(f"Discriminator Stats:")
    print(f"  Real images - Mean score: {real_scores.mean().item():.4f}, "
          f"Std: {real_scores.std().item():.4f}")
    print(f"  Fake images - Mean score: {fake_scores.mean().item():.4f}, "
          f"Std: {fake_scores.std().item():.4f}")

    # Discriminator accuracy
    print("Plotting confusion matrix...")
    plot_confusion_matrix(real_scores, fake_scores, 
                        save_path=os.path.join(args.output_dir, 'confusion_matrix_eval.png'))

    print(f"\nEvaluation complete! Outputs saved to {args.output_dir}.")


if __name__ == '__main__':
    main()