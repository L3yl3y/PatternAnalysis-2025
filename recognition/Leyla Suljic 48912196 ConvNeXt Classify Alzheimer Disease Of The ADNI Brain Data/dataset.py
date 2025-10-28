from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------------------------------------------------------
# This pretty much is just to run through the dataset and load the images and their labels - nothing exciting.
# ------------------------------------------------------------------------------------------------------------------
class ADNIDataset(Dataset):
    def __init__(self, data_root, mode = 'train', transform = None, img_size = 224):
        self.data_root = Path(data_root)
        self.mode = mode
        self.transform = transform
        self.img_size = img_size
        self.samples = self._load_file_paths()
    
    def _load_file_paths(self):
        samples = [] # Empty list which will store both (image path, label) tuples.
        split_dir = self.data_root / self.mode # Kinda gross but this splits directory into train NC/test AD/test NC/train AD.
        
        if not split_dir.exists():
            raise ValueError(f"Directory {split_dir} not found!") # Safety slop.
        
        # Load AD samples - these are of the alzheimers disease - these will get label 1.
        ad_dir = split_dir / 'AD'
        if ad_dir.exists(): # Only processes of the actual AD directory exists --> which it should fyi lmao.
            ad_files = list(ad_dir.glob('*.jpg')) + list(ad_dir.glob('*.jpeg')) # This just finds all .jpg and .jpeg files.
            samples.extend([(str(f), 1) for f in ad_files]) # Would look something this: samples = [('/path/to/ad1.jpg', 1), ('/path/to/ad2.jpg', 1), ...].
        
        # Load NC samples - these are normal brains as like a scientific experiment control group.
        # This is gonna get label 0 btw.
        for folder_name in ['NC', 'CN']: # Same concept above but just different .jpg/.jpeg file brain type.
            normal_dir = split_dir / folder_name
            if normal_dir.exists():
                normal_files = list(normal_dir.glob('*.jpg')) + list(normal_dir.glob('*.jpeg'))
                samples.extend([(str(f), 0) for f in normal_files])
                break
        
        if len(samples) == 0:
            raise ValueError(f"No images found in {split_dir}!") # Not really necessary but safety slop regardless.
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB') # Image open to actually read the jpeg file.
        
        if img.size != (self.img_size, self.img_size):
            # Image.LANCZOS is just a high-quality downsampling filter (apparently its good for medical images).
            img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
        
        if self.transform:
            img = self.transform(img) # Apply the transforms we'll have defined later (in get_transforms()).
        
        return img, label

# ------------------------------------------------------------------------------------------------------------------
# TRANSFORMS!! --> This alters the images slightly to make the model smarter so it has more variety in the data.
# ------------------------------------------------------------------------------------------------------------------
def get_transforms(mode = 'train', img_size = 224):
    # I refuse to have ugly code.
    # Fyi we don't need to have horizontal modifications because the data is already centered perfectly.
    # If anything it would just cause training to take longer which is gross.

    if mode == 'train':
        return transforms.Compose([
            transforms.ColorJitter(brightness = 0.2, contrast = 0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])
    
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])

# ------------------------------------------------------------------------------------------------------------------
# DATALOADER - this is what actually feeds the data into the model we train.
# What this should do is split the training into train + validation and then load the data (our odds are 85%/15%).
# ------------------------------------------------------------------------------------------------------------------
def create_dataloaders(data_root, batch_size = 16, img_size = 224, num_workers = 8, val_split = 0.15):
    # Increased number of workers to 8 for better data transfer (I did that in demo2 it worked well).
    train_full = ADNIDataset(data_root, mode = 'train', transform = None, img_size = img_size)
    
    # Load test data; I know its gross all being in a line like this idgaf.
    test_dataset = ADNIDataset(data_root, mode = 'test', transform = get_transforms('test', img_size), img_size = img_size)
    # Note, the test_dataset is like the final exam and should only be after the model has trained.
    
    # This is where we split the training into those training + testing aspect.
    train_paths = [sample[0] for sample in train_full.samples]
    train_labels = [sample[1] for sample in train_full.samples]
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels,
        test_size = val_split,
        stratify = train_labels,
        random_state = 42
    )
    
    # Train dataset with transforms and augmentations.
    train_dataset = ADNIDataset.__new__(ADNIDataset)
    train_dataset.__dict__.update({
        'data_root': Path(data_root),
        'mode': 'train',
        'transform': get_transforms('train', img_size),
        'img_size': img_size,
        'samples': list(zip(train_paths, train_labels))
    })
    
    # Create validation dataset (no augmentation).
    val_dataset = ADNIDataset.__new__(ADNIDataset)
    val_dataset.__dict__.update({
        'data_root': Path(data_root),
        'mode': 'train',
        'transform': get_transforms('val', img_size),
        'img_size': img_size,
        'samples': list(zip(val_paths, val_labels))
    })
    
    # Print dataset statistics
    print(f"\nTrain: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    
    # These are the three sets of dataloaders fyi so - train, validate, and final test (in this order).
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