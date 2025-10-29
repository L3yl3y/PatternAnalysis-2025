from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------------------------------------------------------
# FIXED: Added much stronger augmentation to prevent overfitting!
# ------------------------------------------------------------------------------------------------------------------
class ADNIDataset(Dataset):
    def __init__(self, data_root, mode = 'train', transform = None, img_size = 224):
        self.data_root = Path(data_root)
        self.mode = mode
        self.transform = transform
        self.img_size = img_size
        self.samples = self._load_file_paths()
    
    def _load_file_paths(self):
        samples = []
        split_dir = self.data_root / self.mode
        
        if not split_dir.exists():
            raise ValueError(f"Directory {split_dir} not found!")
        
        # Load AD samples (label 1)
        ad_dir = split_dir / 'AD'
        if ad_dir.exists():
            ad_files = list(ad_dir.glob('*.jpg')) + list(ad_dir.glob('*.jpeg'))
            samples.extend([(str(f), 1) for f in ad_files])
        
        # Load NC samples (label 0)
        for folder_name in ['NC', 'CN']:
            normal_dir = split_dir / folder_name
            if normal_dir.exists():
                normal_files = list(normal_dir.glob('*.jpg')) + list(normal_dir.glob('*.jpeg'))
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
        
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

# ------------------------------------------------------------------------------------------------------------------
# FIXED TRANSFORMS!! --> Much stronger augmentation to prevent overfitting
# ------------------------------------------------------------------------------------------------------------------
def get_transforms(mode = 'train', img_size = 224):
    if mode == 'train':
        # FIXED: Added rotation, horizontal flip, more aggressive color jitter
        # Medical images benefit from these augmentations!
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),  # NEW: Brains are symmetric
            transforms.RandomRotation(degrees=15),    # NEW: Small rotations are realistic
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # NEW: Small shifts
                scale=(0.9, 1.1)       # NEW: Slight zoom in/out
            ),
            transforms.ColorJitter(
                brightness=0.3,   # Increased from 0.2
                contrast=0.3,     # Increased from 0.2
                saturation=0.2,   # NEW: Add saturation variation
                hue=0.05          # NEW: Tiny hue shift
            ),
            transforms.RandomGrayscale(p=0.1),  # NEW: Sometimes convert to grayscale
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(  # NEW: Randomly erase patches (forces model to use multiple regions)
                p=0.3, 
                scale=(0.02, 0.1), 
                ratio=(0.3, 3.3)
            )
        ])
    
    else:  # val or test
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# ------------------------------------------------------------------------------------------------------------------
# DATALOADER - unchanged, this part was fine
# ------------------------------------------------------------------------------------------------------------------
def create_dataloaders(data_root, batch_size = 16, img_size = 224, num_workers = 8, val_split = 0.15):
    train_full = ADNIDataset(data_root, mode = 'train', transform = None, img_size = img_size)
    test_dataset = ADNIDataset(data_root, mode = 'test', transform = get_transforms('test', img_size), img_size = img_size)
    
    train_paths = [sample[0] for sample in train_full.samples]
    train_labels = [sample[1] for sample in train_full.samples]
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels,
        test_size = val_split,
        stratify = train_labels,
        random_state = 42
    )
    
    # Train dataset with strong augmentation
    train_dataset = ADNIDataset.__new__(ADNIDataset)
    train_dataset.__dict__.update({
        'data_root': Path(data_root),
        'mode': 'train',
        'transform': get_transforms('train', img_size),
        'img_size': img_size,
        'samples': list(zip(train_paths, train_labels))
    })
    
    # Validation dataset (no augmentation)
    val_dataset = ADNIDataset.__new__(ADNIDataset)
    val_dataset.__dict__.update({
        'data_root': Path(data_root),
        'mode': 'train',
        'transform': get_transforms('val', img_size),
        'img_size': img_size,
        'samples': list(zip(val_paths, val_labels))
    })
    
    print(f"\n📊 Dataset loaded:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val:   {len(val_dataset)} samples")
    print(f"   Test:  {len(test_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
        pin_memory = True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = True
    )
    
    return train_loader, val_loader, test_loader