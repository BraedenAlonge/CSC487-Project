import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from models import Generator, Discriminator
from data import PokemonDataset
from utils.reproducibility import set_seed, make_deterministic
from utils.visualization import save_image_grid, plot_training_curves
from utils.metrics import calculate_fid

def parse_args():
    parser = argparse.ArgumentParser(description='Train DCGAN for PokÃ©mon generation')
    parser.add_argument('--config', type=str, default='configs/baseline.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_models(config, device):
    netG = Generator(
        nz=config['model']['nz'],
        ngf=config['model']['ngf'],
        nc=config['model']['nc'],
        kernel_size=config['model'].get('kernel_size', 4),
        stride=config['model'].get('stride', 2),
        padding=config['model'].get('padding', 1),
        dropout=config['model'].get('dropout_g', 0.0)
    ).to(device)
    
    netD = Discriminator(
        nc=config['model']['nc'],
        ndf=config['model']['ndf'],
        use_sigmoid=True,
        kernel_size=config['model'].get('kernel_size', 4),
        stride=config['model'].get('stride', 2),
        padding=config['model'].get('padding', 1),
        dropout=config['model'].get('dropout_d', 0.0)
    ).to(device)
    
    # Initialize weights
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    print(f"Generator parameters: {sum(p.numel() for p in netG.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in netD.parameters()):,}")
    
    return netG, netD

def create_dataloaders(config):
    if 'val_dir' in config['data'] and config['data']['val_dir'] is not None:
        # Use separate directories for train and validation
        train_dataset = PokemonDataset(
            root_dir=config['data']['train_dir'],
            augment=config['data']['augment']
        )
        val_dataset = PokemonDataset(
            root_dir=config['data']['val_dir'],
            augment=False
        )
    else:
        # Split single dataset into train/val
        full_dataset = PokemonDataset(
            root_dir=config['data']['train_dir'],
            augment=False  # Don't augment before splitting
        )
        val_size = int(len(full_dataset) * 0.2)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        # Apply augmentation to training set only
        if config['data']['augment']:
            # Note: We can't directly augment a Subset, so we'll handle this in the DataLoader
            # For now, we'll create a wrapper or just use the subset as-is
            # In practice, you might want to create a custom dataset wrapper
            pass
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader

def train_epoch(netG, netD, train_loader, criterion, optimizerG, optimizerD, 
                device, config, epoch, writer):
    netG.train()
    netD.train()
    
    g_losses = []
    d_losses = []
    d_real_acc = []
    d_fake_acc = []
    
    nz = config['model']['nz']
    label_smoothing = config['training'].get('label_smoothing', 0.0)
    grad_clip_d = config['training'].get('grad_clip_d')
    grad_clip_g = config['training'].get('grad_clip_g')
    label_flip_prob = config['training'].get('label_flip_prob', 0.0)
    
    for batch_idx, real_images in enumerate(train_loader):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        
        # Prepare labels
        # One-sided label smoothing: only smooth real labels, keep fake labels at 0
        one_sided_smoothing = config['training'].get('one_sided_label_smoothing', False)
        
        # BCE: with optional one-sided label smoothing
        if one_sided_smoothing and label_smoothing > 0:
            # Only smooth real labels, fake labels stay at 0
            label_real_d = torch.full((batch_size, 1), 1.0 - label_smoothing, device=device)
            label_fake_d = torch.zeros(batch_size, 1, device=device)
        else:
            # Two-sided smoothing (original behavior)
            label_real_d = torch.full((batch_size, 1), 1.0 - label_smoothing, device=device)
            label_fake_d = torch.full((batch_size, 1), label_smoothing, device=device)
        label_real_g = torch.full((batch_size, 1), 1.0, device=device)
        
        # Noisy label flipping: randomly flip labels with specified probability
        if label_flip_prob > 0.0:
            flip_mask_real = torch.rand(batch_size, 1, device=device) < label_flip_prob
            flip_mask_fake = torch.rand(batch_size, 1, device=device) < label_flip_prob
            # Flip: real becomes fake (0), fake becomes real (1)
            label_real_d = torch.where(flip_mask_real, 
                                      torch.zeros_like(label_real_d), 
                                      label_real_d)
            label_fake_d = torch.where(flip_mask_fake,
                                      torch.ones_like(label_fake_d),
                                      label_fake_d)
        
        # Train Discriminator
        netD.zero_grad()
        
        # Real images
        output_real = netD(real_images)
        errD_real = criterion(output_real, label_real_d)
        errD_real.backward()
        D_x = output_real.mean().item()
        
        # Fake images
        noise = torch.randn(batch_size, nz, device=device)
        fake_images = netG(noise)
        output_fake = netD(fake_images.detach())
        errD_fake = criterion(output_fake, label_fake_d)
        errD_fake.backward()
        D_G_z1 = output_fake.mean().item()
        
        errD = errD_real + errD_fake
        
        # Gradient clipping for discriminator (if enabled)
        if grad_clip_d is not None:
            torch.nn.utils.clip_grad_norm_(netD.parameters(), float(grad_clip_d))
        
        optimizerD.step()
        
        # Train Generator (once per batch, generate new fake images)
        netG.zero_grad()
        noise = torch.randn(batch_size, nz, device=device)
        fake_images = netG(noise)
        output = netD(fake_images)
        errG = criterion(output, label_real_g)
        errG.backward()
        
        # Gradient clipping for generator (if enabled)
        if grad_clip_g is not None:
            torch.nn.utils.clip_grad_norm_(netG.parameters(), float(grad_clip_g))
        
        D_G_z2 = output.mean().item()
        optimizerG.step()
        
        # Stats
        g_losses.append(errG.item())
        d_losses.append(errD.item())
        d_real_acc.append((output_real > 0.5).float().mean().item())
        d_fake_acc.append((output_fake < 0.5).float().mean().item())
        
        # Logging
        if batch_idx % config['training']['log_interval'] == 0:
            print(f'Epoch [{epoch}/{config["training"]["epochs"]}] '
                  f'Batch [{batch_idx}/{len(train_loader)}] '
                  f'Loss_D: {errD.item():.4f} '
                  f'Loss_G: {errG.item():.4f} '
                  f'D(x): {D_x:.4f} '
                  f'D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}')
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/Loss_D', errD.item(), global_step)
            writer.add_scalar('Train/Loss_G', errG.item(), global_step)
            writer.add_scalar('Train/D_x', D_x, global_step)
            writer.add_scalar('Train/D_G_z', D_G_z2, global_step)
    
    return {
        'g_loss': np.mean(g_losses),
        'd_loss': np.mean(d_losses),
        'd_real_acc': np.mean(d_real_acc),
        'd_fake_acc': np.mean(d_fake_acc)
    }

def validate(netG, netD, val_loader, criterion, device, config):
    netG.eval()
    netD.eval()
    
    g_losses = []
    d_losses = []
    real_images_list = []
    fake_images_list = []
    
    nz = config['model']['nz']
    n_samples = config['training'].get('val_samples', 64)
    label_smoothing = config['training'].get('label_smoothing', 0.0)
    one_sided_smoothing = config['training'].get('one_sided_label_smoothing', False)
    
    with torch.no_grad():
        for batch_idx, real_images in enumerate(val_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # Discriminator on real images
            # BCE with optional one-sided smoothing
            if one_sided_smoothing and label_smoothing > 0:
                label_real = torch.full((batch_size, 1), 1.0 - label_smoothing, device=device)
                label_fake = torch.zeros(batch_size, 1, device=device)
            else:
                label_real = torch.full((batch_size, 1), 1.0, device=device)
                label_fake = torch.full((batch_size, 1), 0.0, device=device)
            
            output_real = netD(real_images)
            errD_real = criterion(output_real, label_real)
            
            # Discriminator on fake images
            noise = torch.randn(batch_size, nz, device=device)
            fake_images = netG(noise)
            output_fake = netD(fake_images)
            errD_fake = criterion(output_fake, label_fake)
            
            errD = errD_real + errD_fake
            
            # Generator loss
            label_real = torch.full((batch_size, 1), 1.0, device=device)
            output = netD(fake_images)
            errG = criterion(output, label_real)
            
            g_losses.append(errG.item())
            d_losses.append(errD.item())
            
            # Collect samples for visualization
            if len(real_images_list) < n_samples:
                real_images_list.append(real_images[:min(n_samples - len(real_images_list), batch_size)])
                fake_images_list.append(fake_images[:min(n_samples - len(fake_images_list), batch_size)])
    
    # Concatenate samples
    real_samples = torch.cat(real_images_list, dim=0)[:n_samples]
    fake_samples = torch.cat(fake_images_list, dim=0)[:n_samples]
    
    return {
        'g_loss': np.mean(g_losses),
        'd_loss': np.mean(d_losses),
        'real_samples': real_samples,
        'fake_samples': fake_samples
    }

def save_checkpoint(netG, netD, optimizerG, optimizerD, epoch, metrics, config, filepath):
    checkpoint = {
        'epoch': epoch,
        'netG_state_dict': netG.state_dict(),
        'netD_state_dict': netD.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
        'config': config,
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)
    print(f"Saved checkpoint to {filepath}")

def main():
    # Prepare environment for training
    args = parse_args()
    config = load_config(args.config)
    set_seed(config['training']['seed'])
    if config['training'].get('deterministic', False):
        make_deterministic()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['training']['output_dir'], exist_ok=True)
    os.makedirs(config['training']['log_dir'], exist_ok=True)    
    writer = SummaryWriter(log_dir=config['training']['log_dir'])
    
    # Create models, dataloaders, loss and optimizers
    netG, netD = create_models(config, device)
    train_loader, val_loader = create_dataloaders(config)
    
    # Choose loss function
    criterion = nn.BCELoss()
    print("Using BCE loss")
    beta2 = config['training'].get('beta2', 0.999)
    optimizerG = optim.Adam(netG.parameters(), lr=config['training']['lr_g'], 
                           betas=(config['training']['beta1'], beta2))
    optimizerD = optim.Adam(netD.parameters(), lr=config['training']['lr_d'], 
                           betas=(config['training']['beta1'], beta2))
    
    # Early stopping configuration
    early_stopping_config = config['training'].get('early_stopping', {})
    early_stopping_enabled = early_stopping_config.get('enabled', False)
    early_stopping_patience = early_stopping_config.get('patience', 10)
    early_stopping_min_delta = early_stopping_config.get('min_delta', 0.0)
    epochs_without_improvement = 0
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_fid = float('inf')
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if 'metrics' in checkpoint:
            best_fid = checkpoint['metrics'].get('best_fid', float('inf'))
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    print("Starting training...")
    train_losses_g = []
    train_losses_d = []
    val_losses_g = []
    val_losses_d = []
    val_fids = []
    for epoch in range(start_epoch, config['training']['epochs']):
        # Train and validate one epoch
        train_metrics = train_epoch(netG, netD, train_loader, criterion, 
                                    optimizerG, optimizerD, device, config, epoch, writer)
        train_losses_g.append(train_metrics['g_loss'])
        train_losses_d.append(train_metrics['d_loss'])
        val_metrics = validate(netG, netD, val_loader, criterion, device, config)
        val_losses_g.append(val_metrics['g_loss'])
        val_losses_d.append(val_metrics['d_loss'])
        
        print("Calculating FID...")
        fid = calculate_fid(val_metrics['real_samples'], val_metrics['fake_samples'], device=device)
        val_fids.append(fid)
        
        print(f'Epoch [{epoch}/{config["training"]["epochs"]}] '
              f'Val Loss_D: {val_metrics["d_loss"]:.4f} '
              f'Val Loss_G: {val_metrics["g_loss"]:.4f} '
              f'FID: {fid:.4f}')
        
        # TensorBoard logging
        writer.add_scalar('Val/Loss_D', val_metrics['d_loss'], epoch)
        writer.add_scalar('Val/Loss_G', val_metrics['g_loss'], epoch)
        writer.add_scalar('Val/FID', fid, epoch)
        
        # Save sample images and checkpoints
        if epoch % config['training']['save_interval'] == 0:
            save_image_grid(val_metrics['fake_samples'], 
                          os.path.join(config['training']['output_dir'], f'epoch_{epoch}_fake.png'))
            save_image_grid(val_metrics['real_samples'], 
                          os.path.join(config['training']['output_dir'], f'epoch_{epoch}_real.png'))        
        if epoch % config['training']['checkpoint_interval'] == 0:
            checkpoint_path = os.path.join(config['training']['checkpoint_dir'], 
                                         f'checkpoint_epoch_{epoch}.pt')
            save_checkpoint(netG, netD, optimizerG, optimizerD, epoch, 
                          {'best_fid': best_fid}, config, checkpoint_path)
        
        # Save best model and check for early stopping
        if fid < best_fid - early_stopping_min_delta:
            best_fid = fid
            epochs_without_improvement = 0
            best_path = os.path.join(config['training']['checkpoint_dir'], 'best_model.pt')
            save_checkpoint(netG, netD, optimizerG, optimizerD, epoch, 
                          {'best_fid': best_fid}, config, best_path)
            print(f"New best FID: {best_fid:.4f}")
        else:
            epochs_without_improvement += 1
        
        # Early stopping check
        if early_stopping_enabled and epochs_without_improvement >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epochs_without_improvement} epochs without improvement.")
            print(f"Best FID: {best_fid:.4f} at epoch {epoch - epochs_without_improvement}")
            break
    
    # Save final model
    final_path = os.path.join(config['training']['checkpoint_dir'], 'final_model.pt')
    save_checkpoint(netG, netD, optimizerG, optimizerD, config['training']['epochs'] - 1,
                   {'best_fid': best_fid}, config, final_path)
    
    # Also save as baseline.pt for reproducibility requirement
    baseline_path = os.path.join(config['training']['checkpoint_dir'], 'baseline.pt')
    save_checkpoint(netG, netD, optimizerG, optimizerD, config['training']['epochs'] - 1,
                   {'best_fid': best_fid}, config, baseline_path)
    
    # plot training curves
    plot_training_curves(
        train_losses_g, val_losses_g,
        train_metrics={'D_Loss': train_losses_d},
        val_metrics={'D_Loss': val_losses_d, 'FID': val_fids},
        save_path=os.path.join(config['training']['output_dir'], 'training_curves.png')
    )
    
    writer.close()
    print("Training complete!")


if __name__ == '__main__':
    main()