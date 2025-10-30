from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

# Going through and formally understanding the code as well as putting docs before each function for formality sakes.
# This file will make a personal dataset to read the 2 MRI image types.
# Note this holds both the (image, labels) pairs.
# FYI I think we need doc strings for each of our functions ;-; that's okay we will try make that happen sighhhh.
""" Overall this class is for loading the alzeihmers and normal control MRI images and it handles reading file paths
giving everything these labels, applying transforms, and providing samples for PyTorch DataLoaders."""
class ADNIDataset(Dataset):
    """ADNI 2 - class dataset (with the AD = 1, NC/CN = 0)."""
    def __init__(self, data_root, mode = 'train', transform = None, img_size = 224):
        self.data_root = Path(data_root)
        self.mode = mode # Either the train or testing dataset.
        self.transform = transform # Stores my flips and rotations.
        self.img_size = img_size # Must be 224 x 224.
        self.samples = self._load_file_paths() # This will load all the image paths btw.

    """ This is the function that will actually scan the the data directory and build 
    the list of (image_Path, label) pairs."""
    def _load_file_paths(self):
        samples = []
        split_dir = self.data_root / self.mode # The  folder to look inside (data/ADNI/"train")
        
        if not split_dir.exists():
            raise ValueError(f"Directory {split_dir} not found!")
        
        # Load AD samples (label 1):
        ad_dir = split_dir / 'AD'
        if ad_dir.exists(): # .glob(*.jpg) means to try find every file that ends with .jpg.
            # Also adding jpeg to that list for safety (I am not going through all the images to be sure lmao).
            ad_files = list(ad_dir.glob('*.jpg')) + list(ad_dir.glob('*.jpeg'))
            samples.extend([(str(f), 1) for f in ad_files])
        
        # Load NC samples (label 0) - similar to the above AD:
        for folder_name in ['NC', 'CN']:
            normal_dir = split_dir / folder_name
            if normal_dir.exists():
                normal_files = list(normal_dir.glob('*.jpg')) + list(normal_dir.glob('*.jpeg'))
                samples.extend([(str(f), 0) for f in normal_files])
                break
        
        if len(samples) == 0: # If our above [] is still empty even after both AD and NC, then scream.
            raise ValueError(f"No images found in {split_dir}!")
        
        return samples

    """Returns total number of samples in teh dataset - the length of my [] 'shopping list'. """
    def __len__(self):
        return len(self.samples)

    """So PyTorch DataLoader will call this function and say get me the (image, label) pair at the given index
    FYI this is where the images are loaded, resized, and transformed."""
    def __getitem__(self, idx):
        img_path, label = self.samples[idx] # Getting (file_path, label) from our list at the given idx index.
        img = Image.open(img_path).convert('RGB') # Apparently this is cause models like RGB most (so 3 colour channel).
        
        if img.size != (self.img_size, self.img_size): # Check if image is the right zero.
            # If it is not 224 x 224, then resize it (note LANCZOS is a high-quality resizing method).
            img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
        
        if self.transform: # Check if there are transforms we need to apply and then apply them duh.
            img = self.transform(img)
        
        return img, label

"""Modify to have strong augmentation - this will purportedly prevent my overfitting issue.
Note, torchvision.transforms.Compose will return a pipeline of transformations."""
def get_transforms(mode = 'train', img_size = 224):
    if mode == 'train': # Data augmentation to create fake images and stop the model from its memorising.
        # Below are rotations, flips, and aggressive colour jitter as this will benefit the model's learning.
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.RandomRotation(degrees = 15), # Tilt the picture by 15° or so.
            transforms.RandomAffine(
                degrees = 0,
                translate = (0.1, 0.1), # Scooting the image left, up, down by 10% (small shifts).
                scale = (0.9, 1.1)
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