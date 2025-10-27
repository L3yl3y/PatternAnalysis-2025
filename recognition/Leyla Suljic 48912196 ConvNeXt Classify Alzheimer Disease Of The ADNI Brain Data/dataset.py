# ------------------------------------------------------------------------------------------------------------------
#  ADNI Alzheimer's Disease Classification Dataset Loader
# ------------------------------------------------------------------------------------------------------------------
#  This module handles loading and preprocessing of ADNI brain MRI data for binary classification (Normal vs AD).
#  Supports both 2D slice extraction from 3D volumes and preprocessing optimizations.
# ------------------------------------------------------------------------------------------------------------------
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import glob
import os
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


# ------------------------------------------------------------------------------------------------------------------
# ADNI DATASET CLASS - Handles brain MRI loading and preprocessing
# ------------------------------------------------------------------------------------------------------------------
class ADNIDataset(Dataset):
    """
    ADNI Brain MRI Dataset for Alzheimer's Classification

    This dataset loads JPEG format brain MRI images for binary classification.
    Images should be pre-processed 2D slices from brain MRI scans.

    Labels: 0 = Normal (CN), 1 = Alzheimer's Disease (AD)
    """

    def __init__(self, data_root, mode='train', transform=None, img_size=224):
        """
        Args:
            data_root: Path to ADNI data directory (e.g., /home/groups/comp3710/ADNI)
            mode: 'train', 'val', or 'test'
            transform: Optional transforms to apply
            img_size: Target image size (224 for ConvNeXt default)
        """
        self.data_root = Path(data_root)
        self.mode = mode
        self.transform = transform
        self.img_size = img_size

        # Find all JPEG files organized by class
        self.samples = self._load_file_paths()

        print(f"[{mode.upper()}] Loaded {len(self.samples)} samples")

    def _load_file_paths(self):
        """
        Load file paths and labels from ADNI directory structure.
        Expected structure:
            ADNI/
                train/
                    AD/  (Alzheimer's Disease)
                        patient1.jpg
                    NC/  (Normal Control)
                        patient2.jpg
                test/
                    AD/
                        patient3.jpg
                    NC/
                        patient4.jpg
        """
        samples = []

        # Determine the split directory based on mode
        split_dir = self.data_root / self.mode

        if not split_dir.exists():
            raise ValueError(f"Split directory {split_dir} not found!")

        # Load AD samples (label = 1)
        ad_dir = split_dir / 'AD'
        if ad_dir.exists():
            ad_files = list(ad_dir.glob('*.jpg')) + list(ad_dir.glob('*.jpeg'))  # Match .jpg and .jpeg
            samples.extend([(str(f), 1) for f in ad_files])
            print(f"Found {len(ad_files)} AD samples")

        # Load NC/CN (Normal) samples (label = 0) - check both NC and CN folder names
        nc_dir = split_dir / 'NC'
        cn_dir = split_dir / 'CN'

        normal_dir = nc_dir if nc_dir.exists() else cn_dir
        if normal_dir.exists():
            normal_files = list(normal_dir.glob('*.jpg')) + list(normal_dir.glob('*.jpeg'))
            samples.extend([(str(f), 0) for f in normal_files])
            print(f"Found {len(normal_files)} Normal samples")

        if len(samples) == 0:
            raise ValueError(f"No JPEG files found in {split_dir}. Check directory structure!")

        return samples

    def _load_jpeg(self, jpeg_path):
        """
        Load JPEG image and resize if needed.

        Args:
            jpeg_path: Path to JPEG file

        Returns:
            PIL Image object
        """
        try:
            # Load JPEG image
            img = Image.open(jpeg_path).convert('RGB')

            # Resize to target size if needed
            if img.size != (self.img_size, self.img_size):
                img = img.resize((self.img_size, self.img_size), Image.LANCZOS)

            return img

        except Exception as e:
            print(f"Error loading {jpeg_path}: {e}")
            # Return blank image as fallback
            return Image.new('RGB', (self.img_size, self.img_size), color='black')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load and return a single sample.
        Returns:
            image: Tensor of shape (3, 224, 224)
            label: 0 (Normal) or 1 (AD)
        """
        jpeg_path, label = self.samples[idx]

        # Load the JPEG image
        image = self._load_jpeg(jpeg_path)

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)

        return image, label


# ------------------------------------------------------------------------------------------------------------------
# DATA AUGMENTATION & TRANSFORMS
# ------------------------------------------------------------------------------------------------------------------
def get_transforms(mode='train', img_size=224):
    """
    Get appropriate transforms for train/val/test.

    Training augmentations (to prevent overfitting):
    - Random rotations (brains can be tilted slightly)
    - Random flips (left/right brain symmetry)
    - Random crops (zoom in/out)
    - Color jitter (simulate different scanner intensities)

    Why augmentation matters:
    ADNI has ~400 samples total, which is TINY for deep learning.
    Augmentation artificially expands this by showing the model "variations"
    of each brain scan, helping it generalize better.
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.RandomRotation(15),  # Rotate ±15 degrees
            transforms.RandomHorizontalFlip(p=0.5),  # Flip 50% of the time
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),  # Slight zoom
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Vary intensity
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats (standard)
                                 std=[0.229, 0.224, 0.225])
        ])
    else:  # val/test - no augmentation, just normalize
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


# ------------------------------------------------------------------------------------------------------------------
# DATA LOADER CREATION
# ------------------------------------------------------------------------------------------------------------------
def create_dataloaders(data_root, batch_size=16, img_size=224, num_workers=4, val_split=0.15):
    """
    Create train/val/test dataloaders from pre-split directory structure.

    The data should already be split into train/ and test/ folders.
    We'll create a validation set by splitting from the training data.

    Expected structure:
        ADNI/
            train/
                AD/
                NC/ (or CN/)
            test/
                AD/
                NC/ (or CN/)
    """
    print("=" * 80)
    print("SETTING UP ADNI ALZHEIMER'S DATASET")
    print("=" * 80)

    # Load training data
    print("\nLoading training data...")
    train_full_dataset = ADNIDataset(data_root, mode='train', transform=None, img_size=img_size)

    # Load test data
    print("\nLoading test data...")
    test_dataset = ADNIDataset(data_root, mode='test', transform=get_transforms('test', img_size), img_size=img_size)

    # Split training data into train and validation
    train_samples = train_full_dataset.samples
    train_paths = [s[0] for s in train_samples]
    train_labels = [s[1] for s in train_samples]

    # Stratified split to create validation set
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels,
        test_size=val_split,
        stratify=train_labels,
        random_state=42
    )

    # Create train dataset
    train_dataset = ADNIDataset.__new__(ADNIDataset)
    train_dataset.__dict__.update({
        'data_root': Path(data_root),
        'mode': 'train',
        'transform': get_transforms('train', img_size),
        'img_size': img_size,
        'samples': list(zip(train_paths, train_labels))
    })

    # Create validation dataset
    val_dataset = ADNIDataset.__new__(ADNIDataset)
    val_dataset.__dict__.update({
        'data_root': Path(data_root),
        'mode': 'train',  # Still from train folder
        'transform': get_transforms('val', img_size),
        'img_size': img_size,
        'samples': list(zip(val_paths, val_labels))
    })

    # Print dataset statistics
    total_samples = len(train_dataset) + len(val_dataset) + len(test_dataset)
    print(f"\nDataset Split:")
    print(f"  Train: {len(train_dataset)} samples ({len(train_dataset) / total_samples * 100:.1f}%)")
    print(f"    - AD: {sum(train_labels)} | Normal: {len(train_labels) - sum(train_labels)}")
    print(f"  Val:   {len(val_dataset)} samples ({len(val_dataset) / total_samples * 100:.1f}%)")
    print(f"    - AD: {sum(val_labels)} | Normal: {len(val_labels) - sum(val_labels)}")
    print(f"  Test:  {len(test_dataset)} samples ({len(test_dataset) / total_samples * 100:.1f}%)")
    test_labels = [s[1] for s in test_dataset.samples]
    print(f"    - AD: {sum(test_labels)} | Normal: {len(test_labels) - sum(test_labels)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=True  # Faster GPU transfer
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle val/test
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"\nDataLoader Configuration:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Num Workers: {num_workers}")
    print(f"  Train Batches: {len(train_loader)}")
    print(f"  Val Batches: {len(val_loader)}")
    print(f"  Test Batches: {len(test_loader)}")
    print("=" * 80)

    return train_loader, val_loader, test_loader


# ------------------------------------------------------------------------------------------------------------------
# TESTING CODE - Run this file directly to test data loading
# ------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    print("Testing ADNI Dataset Loader...")

    # Update this path to your ADNI data location!
    DATA_ROOT = './data/ADNI/AD_NC'  # Change this!

    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            data_root=DATA_ROOT,
            batch_size=8,
            img_size=224,
            num_workers=2
        )

        # Test loading a batch
        print("\nTesting batch loading...")
        images, labels = next(iter(train_loader))
        print(f"Batch shape: {images.shape}")  # Should be (8, 3, 224, 224)
        print(f"Labels: {labels}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")

        # Visualize a few samples
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for i in range(8):
            ax = axes[i // 4, i % 4]
            img = images[i].permute(1, 2, 0).numpy()
            # Denormalize for display
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.set_title(f"Label: {'AD' if labels[i] == 1 else 'Normal'}")
            ax.axis('off')
        plt.tight_layout()

        # Save as JPEG with high quality (95%)
        plt.savefig('dataset_test_samples.jpeg', format='jpeg', dpi=150, quality=95)
        print("✓ Saved visualization to 'dataset_test_samples.jpeg'")

        print("\n✓ Dataset loader working perfectly!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("Make sure DATA_ROOT points to your ADNI directory!")