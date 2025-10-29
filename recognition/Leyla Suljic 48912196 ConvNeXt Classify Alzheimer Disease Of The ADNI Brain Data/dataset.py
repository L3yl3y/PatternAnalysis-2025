from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import numpy as np
import cv2

# ------------------------------------------------------------------------------------------------------------------
# NEW: CLAHE preprocessing inspired by the paper
# ------------------------------------------------------------------------------------------------------------------
class CLAHETransform:
    """
    Contrast Limited Adaptive Histogram Equalization
    Improves local contrast and highlights subtle features
    Critical for medical imaging!
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
    
    def __call__(self, img):
        # Convert PIL to numpy
        img_np = np.array(img)
        
        # Apply CLAHE to each channel
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        
        if len(img_np.shape) == 2:  # Grayscale
            img_np = clahe.apply(img_np)
        else:  # RGB
            # Convert to LAB color space for better results
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])  # Apply only to L channel
            img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(img_np)


# ------------------------------------------------------------------------------------------------------------------
# Enhanced Dataset with CLAHE option
# ------------------------------------------------------------------------------------------------------------------
class ADNIDataset(Dataset):
    def __init__(self, data_root, mode='train', transform=None, img_size=224, use_clahe=True):
        self.data_root = Path(data_root)
        self.mode = mode
        self.transform = transform
        self.img_size = img_size
        self.use_clahe = use_clahe
        self.samples = self._load_file_paths()
    
    def _load_file_paths(self):
        samples = []
        split_dir = self.data_root / self.mode
        
        if not split_dir.exists():
            raise ValueError(f"Directory {split_dir} not found!")
        
        # Load AD samples (label 1)
        ad_dir = split_dir / 'AD'
        if ad_dir.exists():
            ad_files = list(ad_dir.glob('*.jpg')) + list(ad_dir.glob('*.jpeg')) + list(ad_dir.glob('*.png'))
            samples.extend([(str(f), 1) for f in ad_files])
        
        # Load NC samples (label 0)
        for folder_name in ['NC', 'CN', 'Normal']:
            normal_dir = split_dir / folder_name
            if normal_dir.exists():
                normal_files = list(normal_dir.glob('*.jpg')) + list(normal_dir.glob('*.jpeg')) + list(normal_dir.glob('*.png'))
                samples.extend([(str(f), 0) for f in normal_files])
                break
        
        if len(samples) == 0:
            raise ValueError(f"No images found in {split_dir}!")
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        # Resize first
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
        
        # Apply CLAHE preprocessing if enabled (BEFORE augmentation)
        # This is KEY from the paper!
        if self.use_clahe and self.mode == 'train':
            clahe_transform = CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8))
            img = clahe_transform(img)
        
        # Apply augmentation/normalization transforms
        if self.transform:
            img = self.transform(img)
        
        return img, label


# ------------------------------------------------------------------------------------------------------------------
# Enhanced transforms with paper's strategy
# ------------------------------------------------------------------------------------------------------------------
def get_transforms(mode='train', img_size=224):
    if mode == 'train':
        # Strong augmentation matching the paper's strategy
        return transforms.Compose([
            # Geometric augmentations
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=5
            ),
            
            # Color augmentations (more aggressive for medical images)
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.05
            ),
            
            # Occasionally convert to grayscale (forces robust features)
            transforms.RandomGrayscale(p=0.1),
            
            # Advanced augmentations
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.3),
            
            # Convert to tensor and normalize
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            
            # Random erasing (cutout) - forces model to use multiple regions
            transforms.RandomErasing(
                p=0.3,
                scale=(0.02, 0.15),
                ratio=(0.3, 3.3),
                value='random'
            )
        ])
    
    else:  # val or test
        # No augmentation, just normalization
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


# ------------------------------------------------------------------------------------------------------------------
# Dataloader creation
# ------------------------------------------------------------------------------------------------------------------
def create_dataloaders(data_root, batch_size=16, img_size=224, num_workers=8, val_split=0.15, use_clahe=True):
    """
    Creates train, val, and test dataloaders
    
    Args:
        use_clahe: Whether to use CLAHE preprocessing (recommended for medical images)
    """
    train_full = ADNIDataset(data_root, mode='train', transform=None, img_size=img_size, use_clahe=False)
    test_dataset = ADNIDataset(data_root, mode='test', transform=get_transforms('test', img_size), 
                               img_size=img_size, use_clahe=False)
    
    train_paths = [sample[0] for sample in train_full.samples]
    train_labels = [sample[1] for sample in train_full.samples]
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels,
        test_size=val_split,
        stratify=train_labels,
        random_state=42
    )
    
    # Train dataset with CLAHE + strong augmentation
    train_dataset = ADNIDataset.__new__(ADNIDataset)
    train_dataset.__dict__.update({
        'data_root': Path(data_root),
        'mode': 'train',
        'transform': get_transforms('train', img_size),
        'img_size': img_size,
        'use_clahe': use_clahe,
        'samples': list(zip(train_paths, train_labels))
    })
    
    # Validation dataset (no augmentation, no CLAHE)
    val_dataset = ADNIDataset.__new__(ADNIDataset)
    val_dataset.__dict__.update({
        'data_root': Path(data_root),
        'mode': 'train',
        'transform': get_transforms('val', img_size),
        'img_size': img_size,
        'use_clahe': False,  # No CLAHE for validation
        'samples': list(zip(val_paths, val_labels))
    })
    
    print(f"\n📊 Dataset loaded:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val:   {len(val_dataset)} samples")
    print(f"   Test:  {len(test_dataset)} samples")
    if use_clahe:
        print(f"   🔬 CLAHE preprocessing: ENABLED (like the paper!)")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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
    
    return train_loader, val_loader, test_loader