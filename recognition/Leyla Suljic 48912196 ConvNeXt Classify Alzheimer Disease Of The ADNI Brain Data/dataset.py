"""
Dataset loader for ADNI Alzheimer's Disease classification
Includes preprocessing optimized for medical brain MRI images
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import numpy as np
import cv2


class CLAHETransform:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization)
    Essential for enhancing brain structure visibility in MRI
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
    
    def __call__(self, img):
        img_np = np.array(img)
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, 
            tileGridSize=self.tile_grid_size
        )
        
        if len(img_np.shape) == 2:  # Grayscale
            img_np = clahe.apply(img_np)
        else:  # RGB
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(img_np)


class ADNIDataset(Dataset):
    """
    ADNI Dataset for Alzheimer's Disease Classification
    Loads AD (Alzheimer's Disease) and NC (Normal Control) brain MRIs
    """
    def __init__(self, data_root, mode='train', transform=None, 
                 img_size=224, use_clahe=True):
        """
        Args:
            data_root: Root directory containing train/test folders
            mode: 'train', 'val', or 'test'
            transform: Torchvision transforms
            img_size: Target image size
            use_clahe: Apply CLAHE preprocessing
        """
        self.data_root = Path(data_root)
        self.mode = mode
        self.transform = transform
        self.img_size = img_size
        self.use_clahe = use_clahe
        self.samples = self._load_file_paths()
        
        print(f"   Loaded {mode} set: {len(self.samples)} samples")
    
    def _load_file_paths(self):
        """Load all image paths with labels"""
        samples = []
        
        # Determine directory based on mode
        if self.mode in ['train', 'val']:
            split_dir = self.data_root / 'train'
        else:
            split_dir = self.data_root / 'test'
        
        if not split_dir.exists():
            raise ValueError(f"Directory {split_dir} not found!")
        
        # Load AD samples (label 1)
        ad_dir = split_dir / 'AD'
        if ad_dir.exists():
            ad_files = (list(ad_dir.glob('*.jpg')) + 
                       list(ad_dir.glob('*.jpeg')) + 
                       list(ad_dir.glob('*.png')))
            samples.extend([(str(f), 1) for f in ad_files])
        
        # Load NC/CN samples (label 0)
        for folder_name in ['NC', 'CN', 'Normal']:
            normal_dir = split_dir / folder_name
            if normal_dir.exists():
                normal_files = (list(normal_dir.glob('*.jpg')) + 
                               list(normal_dir.glob('*.jpeg')) + 
                               list(normal_dir.glob('*.png')))
                samples.extend([(str(f), 0) for f in normal_files])
                break
        
        if len(samples) == 0:
            raise ValueError(f"No images found in {split_dir}!")
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image as grayscale
        img = Image.open(img_path).convert('L')
        
        # Resize if needed
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
        
        # Apply CLAHE if enabled (only during training)
        if self.use_clahe and self.mode == 'train':
            clahe_transform = CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8))
            img = clahe_transform(img)
        
        # Apply augmentation transforms
        if self.transform:
            img = self.transform(img)
        
        return img, label


def get_transforms(mode='train', img_size=224):
    """
    Get appropriate transforms for each mode
    Training includes aggressive augmentation
    """
    if mode == 'train':
        return transforms.Compose([
            # Geometric augmentations
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=5
            ),
            
            # Convert to tensor (will be single channel)
            transforms.ToTensor(),
            
            # Normalize for grayscale (mean and std for single channel)
            transforms.Normalize(mean=[0.5], std=[0.5]),
            
            # Random erasing (cutout augmentation)
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
        ])
    else:
        # Validation/Test: no augmentation
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])


def create_dataloaders(data_root, batch_size=32, img_size=224, 
                      num_workers=4, val_split=0.1, use_clahe=True):
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_root: Path to ADNI/AD_NC directory
        batch_size: Batch size for training
        img_size: Image size (224 recommended for GFNet)
        num_workers: Number of data loading workers
        val_split: Validation split ratio (0.1 = 90/10 train/val)
        use_clahe: Apply CLAHE preprocessing
    
    Returns:
        train_loader, val_loader, test_loader
    """
    print("\n" + "="*70)
    print("LOADING ADNI DATASET")
    print("="*70)
    print(f"   Settings:")
    print(f"   • Data root: {data_root}")
    print(f"   • Image size: {img_size}x{img_size}")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Train/Val split: {int((1-val_split)*100)}/{int(val_split*100)}")
    print(f"   • CLAHE preprocessing: {'Enabled' if use_clahe else 'Disabled'}")
    
    # Load full training dataset
    train_full = ADNIDataset(
        data_root, mode='train', 
        transform=None,  # Will add later
        img_size=img_size, 
        use_clahe=False  # Will apply per split
    )
    
    # Load test dataset
    test_dataset = ADNIDataset(
        data_root, mode='test',
        transform=get_transforms('test', img_size),
        img_size=img_size,
        use_clahe=False
    )
    
    # Split train into train/val (stratified)
    train_paths = [s[0] for s in train_full.samples]
    train_labels = [s[1] for s in train_full.samples]
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels,
        test_size=val_split,
        stratify=train_labels,
        random_state=42
    )
    
    # Create train dataset with augmentation
    train_dataset = ADNIDataset.__new__(ADNIDataset)
    train_dataset.__dict__.update({
        'data_root': Path(data_root),
        'mode': 'train',
        'transform': get_transforms('train', img_size),
        'img_size': img_size,
        'use_clahe': use_clahe,
        'samples': list(zip(train_paths, train_labels))
    })
    
    # Create validation dataset (no augmentation)
    val_dataset = ADNIDataset.__new__(ADNIDataset)
    val_dataset.__dict__.update({
        'data_root': Path(data_root),
        'mode': 'val',
        'transform': get_transforms('val', img_size),
        'img_size': img_size,
        'use_clahe': False,
        'samples': list(zip(val_paths, val_labels))
    })
    
    print(f"\n   Dataset Statistics:")
    print(f"   • Train: {len(train_dataset.samples)} samples")
    
    # Count class distribution
    train_ad = sum(1 for _, label in train_dataset.samples if label == 1)
    train_nc = len(train_dataset.samples) - train_ad
    print(f"     - AD: {train_ad}, NC: {train_nc}")
    
    print(f"   • Validation: {len(val_dataset.samples)} samples")
    val_ad = sum(1 for _, label in val_dataset.samples if label == 1)
    val_nc = len(val_dataset.samples) - val_ad
    print(f"     - AD: {val_ad}, NC: {val_nc}")
    
    print(f"   • Test: {len(test_dataset.samples)} samples")
    
    # Create balanced sampler for training
    train_labels_array = [label for _, label in train_dataset.samples]
    class_counts = np.bincount(train_labels_array)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_labels_array]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    print("="*70 + "\n")
    
    return train_loader, val_loader, test_loader