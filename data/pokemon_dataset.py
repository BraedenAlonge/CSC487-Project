import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class PokemonDataset(Dataset):    
    def __init__(self, root_dir, transform=None, augment=False, aug_config=None):
        self.root_dir = root_dir
        self.image_paths = []
        self.aug_config = aug_config or {}
        
        # Collect all image paths
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root, file))
        
        print(f'Found {len(self.image_paths)} images in {root_dir}')

        # Base transforms: resize, convert RGBA to RGB, normalize to [-1, 1]
        base_transforms = [
            transforms.Resize((64, 64)),
            transforms.Lambda(self._rgba_to_rgb),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ]
        
        # Add augmentation if requested
        if augment:
            # Get augmentation parameters from config
            hflip_prob = self.aug_config.get('horizontal_flip_prob', 0.5)
            vflip_prob = self.aug_config.get('vertical_flip_prob', 0.5)
            rotation_deg = self.aug_config.get('rotation_degrees', 5)
            
            aug_transforms = [
                transforms.RandomHorizontalFlip(p=hflip_prob),
                transforms.RandomVerticalFlip(p=vflip_prob),
                transforms.RandomRotation(degrees=rotation_deg),
            ]
            
            self.transform = transforms.Compose(aug_transforms + base_transforms)
        else:
            self.transform = transforms.Compose(base_transforms)

        # Override with custom transform if provided
        if transform:
            self.transform = transform

    def _rgba_to_rgb(self, img):
        if img.mode == 'RGBA':
            bg = Image.new('RGB', img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])  # Use alpha channel as mask
            return bg
        return img.convert('RGB')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path)
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            return torch.zeros(3, 64, 64)