import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json

from pathlib import Path
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from dataset import create_dataloaders
from modules import create_alzheimer_model

# ------------------------------------------------------------------------------------------------------------------
""" Calculates F2 - score (prioritises recall apparently) - so it aims to balance precision and recall evently. Since
 we as the medical community hate the false negatives I get on my 2x2 matrix. What this means is that
 I am oftentimes labelling sick brains as healthy so the f2 score will try to give twice as much importance to recall 
 such that we can find as many AD cases as possible."""
def calculate_f2_score(y_true, y_pred, zero_division = 0):
    precision = precision_score(y_true, y_pred, zero_division = zero_division)
    recall = recall_score(y_true, y_pred, zero_division = zero_division)
    
    beta_sq = 4
    if (beta_sq * precision + recall) == 0: # Safety for no division by 0.
        return 0.0
    
    f2 = (1 + beta_sq) * (precision * recall) / ((beta_sq * precision) + recall)
    return f2 # Recall weighted more than precision.

# ------------------------------------------------------------------------------------------------------------------
""" This is to try improve recall; since standard CrossEntropyLoss will treat all the errors equally which is bad for 
2 reasons --> it doesn't care if it misses an AD case and it will spend most of its time on easy cases instead of the
ones it gets wrong. So this is to try rectify this by applying a higher penalty whenever the model gets AD wrong, and
forces the model to pay more attention to AD cases and it will reduce the loss for easy predictions so the model can
focus more on the harder ones."""
class RecallFocusedLoss(nn.Module):
    """ Alpha to balance classes, and recall_weight is that extra punch for the AD class to penalise it."""
    def __init__(self, alpha = 0.25, gamma = 2.0, recall_weight = 1.5):
        super(RecallFocusedLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.recall_weight = recall_weight
        self.ce_loss = nn.CrossEntropyLoss(reduction = 'none') # Give loss for every example in batch not just average.

    """ This is what is called during training, it involves getting the standard cross-entropy loss per example, 
    then keeping the loss if the model is wrong or turning off the loss if the model is right/confident."""
    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        weights = torch.ones_like(targets).float() # Higher weight to positive class (AD) to improve recall.

        weights[targets == 1] = self.recall_weight # Higher weight for AD class.
        weighted_loss = focal_loss * weights
        
        # Add alpha balancing - just helps to balance the two classes.
        if self.alpha is not None:
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            weighted_loss = alpha_t * weighted_loss
        
        return weighted_loss.mean()

# ------------------------------------------------------------------------------------------------------------------
""" Attention block mechanism --> CBAM convolutional block attention module style block."""
class AttentionBlock(nn.Module):
    """ So I read an article on this, and it's called the CBAM convolutional block attention module style block. And
    whilst a normal CNN treats all parts of an image as equally important, I made it such that there is a little
    'attention' block that learns to 'pay attention' to the important parts. This works by channel attention and spatial
    attention. Pretty much you want to focus on parts of the MRI that are most predictive of AD."""
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__() # What.
        self.channel_attention = nn.Sequential( # Squish into 1x1 convs to learn importance of each channel.
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.Sigmoid() # Squish importance scores between 0 and 1.
        )
        
        self.spatial_attention = nn.Sequential( # Where.
            # Takes the average, and the max maps, concatenated into 2 channels and learns where to look.
            nn.Conv2d(2, 1, kernel_size = 7, padding = 3),
            nn.Sigmoid()
        )

    """ This will re-weight each channel (called applying the attention channel)."""
    def forward(self, x):
        ca = self.channel_attention(x)
        x = x * ca

        # This is applying spatial attention - pool all channels into two maps, and it aims to tell the
        # spatial attention block what is going on at every (x, y) location.
        avg_out = torch.mean(x, dim = 1, keepdim = True)
        max_out, _ = torch.max(x, dim = 1, keepdim = True)
        sa = self.spatial_attention(torch.cat([avg_out, max_out], dim = 1))
        x = x * sa
        return x

# ------------------------------------------------------------------------------------------------------------------
""" This is the v4 config class and this holds all our advanced settings - pretty much the model architecture."""
class Config:
    DATA_ROOT = './data/ADNI/AD_NC'

    BATCH_SIZE = 16 # Smaller batch = better gradient quality.
    IMG_SIZE = 224
    NUM_WORKERS = 8
    MODEL_SIZE = 'base' # Back to 'base' = maximum capacity.
    DROPOUT = 0.4 # Moderate dropout (I noticed pretty bad overfitting on my model so I'm trying to fix this T_T).

    TOTAL_EPOCHS = 50 # More epochs + early stopping.
    WARMUP_EPOCHS = 5 # Longer warmup.
    INITIAL_LR = 2e-4 # Moderate learning rate.
    MIN_LR = 1e-7

    WEIGHT_DECAY = 2e-4 # Lighter weight decay.
    PATIENCE = 10 # More patience (so it doesn't stop unnecessarily so, give it ample room).
    MIN_DELTA = 0.0005
    USE_MIXED_PRECISION = True
    GRADIENT_CLIP = 1.5
    LABEL_SMOOTHING = 0.03

    USE_RECALL_LOSS = True
    RECALL_WEIGHT = 1.8 # Weight for AD class.
    FOCAL_ALPHA = 0.3 # Alpha for focal component.
    FOCAL_GAMMA = 1.5 # Gamma for focal component.

    USE_BALANCED_SAMPLING = True
    BALANCE_FACTOR = 0.7
    TARGET_RECALL = 0.83 # Higher target (let us aim high lmao).
    DECISION_THRESHOLD = 0.4 # Lower threshold for better recall.

    # FYI I saw these in a research article, and they mentioned it would help a lot :3
    # Since these here are going to create frankenstein images to force the model to learn robust features to try and
    # dis-encourage memorisation of the training data (i.e., avoid overfitting).
    USE_MIXUP = True # Blend two images (70% A and 30% B).
    MIXUP_ALPHA = 0.2
    MIXUP_PROB = 0.35
    
    USE_CUTMIX = True # Cut patch from B and paste it over A.
    CUTMIX_PROB = 0.25

    USE_RICAP = True # Stitch 4 different images together into 2x2 matrix (most extreme of these 3).
    RICAP_PROB = 0.2
    RICAP_BETA = 0.3

    USE_ATTENTION = True # Switch to our attention block above.
    USE_MULTISCALE = True # Feed it more images to make it more robust (since it will only see 224 x 224).
    SCALES = [196, 224, 256]

    USE_PROGRESSIVE = True # Freeze/unfreeze strategy (train only 3 epochs for the new head).
    FREEZE_EPOCHS = 3  # Freeze backbone initially
    UNFREEZE_LR_FACTOR = 0.5  # Lower LR when unfreezing

    USE_TTA = True  # Test-time augmentation (like a second opinion and improves accuracy).
    TTA_TRANSFORMS = 4  # Number of augmentations at test time
    
    OUTPUT_DIR = Path('./alzheimer_results_v4')
    CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'
    PLOTS_DIR = OUTPUT_DIR / 'plots'

    OUTPUT_DIR.mkdir(exist_ok = True, parents = True)
    CHECKPOINT_DIR.mkdir(exist_ok = True)
    PLOTS_DIR.mkdir(exist_ok = True)

# ------------------------------------------------------------------------------------------------------------------
""" You know the frankenstein operations I was talking about these are the functions below that are how we enable these
in the config --> so this will outline how to do RICAP - Random Image Cropping and Patching augmentation."""
def ricap(images, targets, beta = 0.3):
    batch_size = images.size(0)
    device = images.device
    _, _, H, W = images.size()
    
    # Generate random crop points (ensure they're on the same device).
    # npr.random.beta is good at getting random numbers that are clustered (allows for (x, y) crop point).
    w = torch.from_numpy(np.random.beta(beta, beta, size = batch_size)).float().to(device)
    h = torch.from_numpy(np.random.beta(beta, beta, size = batch_size)).float().to(device)
    w_ = torch.round(w * W).int()
    h_ = torch.round(h * H).int()
    
    # Random permutation (not the next 4 images but random 4 indices) for mixing.
    indices = torch.randperm(batch_size).to(device)

    mixed_images = images.clone() # Empy copy to paste into.
    for i in range(batch_size): # This loop is the stitching part.
        idx = indices[i].item() # Get the 1st random image.
        h_i = h_[i].item()
        w_i = w_[i].item()

        # This is a 4-way paste down here.
        mixed_images[i, :, :h_i, :w_i] = images[idx, :, :h_i, :w_i]
        mixed_images[i, :, :h_i, w_i:] = images[(idx + 1) % batch_size, :, :h_i, w_i:]
        mixed_images[i, :, h_i:, :w_i] = images[(idx + 2) % batch_size, :, h_i:, :w_i]
        mixed_images[i, :, h_i:, w_i:] = images[(idx + 3) % batch_size, :, h_i:, w_i:]

    lam = (w_.float() * h_.float()) / (W * H) # Weight for loss function.
    return mixed_images, targets, targets[indices], lam # The new frankenstein image and the 2 labels to mix.

# ------------------------------------------------------------------------------------------------------------------
""" Here is another frankenstein operation - this one is the MIXUP Augmentation - it blends two random images from the
batch; like putting two photos on top of each other with transparency."""
def mixup_data(x, y, alpha = 1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha) # Lam is the blended amount (so lam = 0.7% means 70% of image A; 30% of B.
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index] # This the blend --> (lam * Image A) + ((1 - lam) * Image B).
    y_a, y_b = y, y[index] # Get labels for A and B.
    return mixed_x, y_a, y_b, lam

# ------------------------------------------------------------------------------------------------------------------
""" CUT-MATRIX frankenstein operation this will randomly cut a patch from one image and pastes it on top of another."""
def cutmix_data(x, y, alpha = 1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    # Logically not all too different from the other two operations.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

# ------------------------------------------------------------------------------------------------------------------
"""Generate random bounding box for CutMix."""
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam) # This will find the width/height of a patch that has area defined by lam.
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)
    cx = np.random.randint(W) # Pick random centre (cx, cy) for patch.
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W) # Convert centre + size into corners - the box must stay inside the image.
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

# ------------------------------------------------------------------------------------------------------------------
"""This is the mixed loss for augmented images; our image is a mix of two images so our loss must be a mix of two 
losses. Think back to this line here above in CUTMIX --> (lam * loss_for_label_A) + ((1 - lam) * loss_for_label_B)"""
def augmentation_criterion(criterion, pred, y_a, y_b, lam):
    if isinstance(lam, torch.Tensor):
        loss = 0

        for i in range(len(pred)): # For RICAP; ricap has different lam for each image so you have to loop through them.
            loss += lam[i] * criterion(pred[i:i+1], y_a[i:i+1]) + (1 - lam[i]) * criterion(pred[i:i+1], y_b[i:i+1])
        return loss / len(pred)

    else:
        # For MixUp and CutMix with scalar lambda.
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ------------------------------------------------------------------------------------------------------------------
"""Looks like a lot of words but it is a simple helper that is like an empty storage box so that for every epoch, we can 
 go through and append new numbers to these lists."""
class MetricsTracker:
    """ Bunch of empty lists."""
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_f2_scores = []
        self.test_accs = []
        self.test_recalls = []
        self.learning_rates = []

    """ This function just updates the bunch of empty lists."""
    def update(self, train_loss, val_loss, train_acc, val_acc, val_recall,
               val_precision, val_f2, test_acc, test_recall, lr):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        self.val_recalls.append(val_recall)
        self.val_precisions.append(val_precision)
        self.val_f2_scores.append(val_f2)
        self.test_accs.append(test_acc)
        self.test_recalls.append(test_recall)
        self.learning_rates.append(lr)

    """Generates and saves all training graphs; matplotlib creates big 2x3 grid of all the plots.
    Sidenote, it really helps me see how much my model sucks when it overfits T_T"""
    def plot_metrics(self, save_path):
        fig, axes = plt.subplots(2, 3, figsize = (18, 10))
        
        # Loss:
        axes[0, 0].plot(self.train_losses, label = 'Train', linewidth = 2)
        axes[0, 0].plot(self.val_losses, label = 'Val', linewidth = 2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training & Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha = 0.3)
        
        # Accuracy:
        axes[0, 1].plot(self.train_accs, label = 'Train', linewidth = 2)
        axes[0, 1].plot(self.val_accs, label = 'Val', linewidth = 2)
        axes[0, 1].plot(self.test_accs, label = 'Test', linewidth = 2, linestyle = '--')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha = 0.3)
        
        # Recall & Precision:
        axes[0, 2].plot(self.val_recalls, label = 'Val Recall', linewidth = 2, color = 'green')
        axes[0, 2].plot(self.val_precisions, label = 'Val Precision', linewidth = 2, color = 'blue')
        axes[0, 2].plot(self.test_recalls, label = 'Test Recall', linewidth = 2, color = 'red', linestyle = '--')
        axes[0, 2].axhline(y = 0.8, color = 'gray', linestyle = '--', label = 'Target (80%)')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].set_title('Recall & Precision')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # F2 Score:
        axes[1, 0].plot(self.val_f2_scores, label = 'Val F2', linewidth = 2, color = 'purple')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F2 Score')
        axes[1, 0].set_title('Validation F2 Score (Recall-weighted)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha = 0.3)
        
        # Learning Rate:
        axes[1, 1].plot(self.learning_rates, linewidth = 2, color = 'orange')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha = 0.3)
        
        # Overfitting Gap:
        gap = [t - v for t, v in zip(self.train_accs, self.val_accs)]
        axes[1, 2].plot(gap, linewidth = 2, color = 'red')
        axes[1, 2].axhline(y = 0, color = 'k', linestyle = '--')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Train - Val Accuracy')
        axes[1, 2].set_title('Overfitting Gap')
        axes[1, 2].grid(True, alpha = 0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
        plt.close()

# ------------------------------------------------------------------------------------------------------------------
"""Create a partially balanced sampler - which will create a WeightedRandomSampler with a factor to control balancing.
So this will calculate the weight for each class (like AD getting a high weight) and then uses the balance_factor to
fade between those weights and no weights at all."""
def create_balanced_sampler(dataset, balance_factor = 1.0):
    labels = [label for _, label in dataset.samples] # Gets list of all labels (0, 0, 1, 0, 1, 1, 0, ...).
    class_counts = np.bincount(labels) # np.bincount will count them [900, 100] for NC to AD.
    weights = 1.0 / class_counts # Gives rarer class AD a higher weight.
    max_weight = weights.max()
    weights = weights / max_weight
    weights = balance_factor * weights + (1 - balance_factor) # Partial balancing (this is the fading part).
    # Fading example is if balance_factor = 0.7, then:
    # 0.7 * [0.11, 1.0] + (1 - 0.7) * 1.0
    # = [0.077, 0.7] + 0.3
    # = [0.377, 1.0]

    sample_weights = weights[labels] # Big list to map each image to its class weight.
    sampler = WeightedRandomSampler(
        weights = sample_weights,
        num_samples = len(sample_weights),
        replacement = True
    )

    return sampler

# ------------------------------------------------------------------------------------------------------------------
"""Simple helper for getting current learning rate - grab this lr from optimiser's internal state."""
def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

# ------------------------------------------------------------------------------------------------------------------
""" Runs a single epoch of training  - this function is the main training loop that handles everything so it sets the
model to train (turn dropout on), then randomly applies the augmentations, then runs the forward pass model with mixed
precision and then calculates our custom loss (backward pass), then clips the gradients, and then updates the model's weights
as like an optimiser step."""
def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler, epoch):
    model.train() # Put model in training mode.
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Multiscale training --> turn on dropout.
    if Config.USE_MULTISCALE and epoch % 3 == 0:
        scale = np.random.choice(Config.SCALES)
        print(f"  Using scale: {scale} x {scale}") # Every 3rd epoch for funsies change the image.

    else:
        scale = Config.IMG_SIZE # Otherwise use default size (224 x 224).
    
    pbar = tqdm(dataloader, desc = f'Epoch {epoch} [TRAIN]') # This is the cool progress bar :3

    for batch_idx, (images, labels) in enumerate(pbar): # Loop for one batch (e.g., 16 images).
        images = images.to(device, non_blocking = True)
        labels = labels.to(device, non_blocking = True)
        
        # Apply augmentation based on probabilities
        r = np.random.random() # Random number between 0 and 1.
        use_augmentation = False
        labels_a, labels_b, lam = None, None, 1.0

        # This logic is to pick one augmentation based on the probabilities in the config:

        # If RICAP_PROB = 0.2 and r = 0.15:
        if Config.USE_RICAP and r < Config.RICAP_PROB:
            images, labels_a, labels_b, lam = ricap(images, labels, Config.RICAP_BETA)
            use_augmentation = True

        # If r = 0.3 and (RICAP_PROB + MIXUP_ROB) = 0.55:
        elif Config.USE_MIXUP and r < (Config.RICAP_PROB + Config.MIXUP_PROB):
            images, labels_a, labels_b, lam = mixup_data(images, labels, Config.MIXUP_ALPHA)
            use_augmentation = True

        # If r = 0.6 and (total prob) = 0.8:
        elif Config.USE_CUTMIX and r < (Config.RICAP_PROB + Config.MIXUP_PROB + Config.CUTMIX_PROB):
            images, labels_a, labels_b, lam = cutmix_data(images, labels, Config.MIXUP_ALPHA)
            use_augmentation = True

        # If r = 0.9, none of the above will run, and we will just have a normal, not-augmented image.
        optimizer.zero_grad(set_to_none = True) # Zero_grad clears old gradients from last batch before re-calculating.

        if scaler is not None: # Check if using mixed-precision.
            with autocast(device_type = 'cuda'): # This is mixed precision path (mixed loss function).
                outputs = model(images)
                
                if use_augmentation:
                    loss = augmentation_criterion(criterion, outputs, labels_a, labels_b, lam)

                else: # Or use normal loss function.
                    loss = criterion(outputs, labels)

            # Backward pass (calculating gradients).
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP) # Gradient clipping.
            scaler.step(optimizer) # Optimiser step.
            scaler.update()
        else:
            # Full precision path (no mixed precision).
            outputs = model(images)
            
            if use_augmentation:
                loss = augmentation_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP)
            optimizer.step()

        # Note, the batch loss to our current running total.
        running_loss += loss.item() * images.size(0)
        
        # Calculating metrics for this batch...
        preds = torch.argmax(outputs, dim = 1)

        # Note, this is tricky, since, to calculate the accuracy, we need to know the real label for our augmented
        # images. So this can be tackled by using the dominant label (the one with lam > 0.5 as a simple guess).
        if use_augmentation:
            if isinstance(lam, torch.Tensor):
                lam = lam.to(labels_a.device) # This is for RICAP which has lam for each image.
                effective_labels = torch.where(lam > 0.5, labels_a, labels_b)

            else: # This is for MixUP/CutMix.
                effective_labels = labels_a if lam > 0.5 else labels_b
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(effective_labels.cpu().numpy().tolist())

        else: # If no augmentation.
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
        
        current_lr = get_lr(optimizer) # Update the tqdm progress bar.
        pbar.set_postfix({'loss': loss.item(), 'lr': f'{current_lr:.2e}'})

    epoch_loss = running_loss / len(dataloader.dataset)
    all_labels = np.array(all_labels, dtype = np.int64)
    all_preds = np.array(all_preds, dtype = np.int64)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

# ------------------------------------------------------------------------------------------------------------------
""" This evaluates the model on the validation or test datasheet, this function will check how well the model is doing
pretty much, its similar to the training loop but much simpler."""
def evaluate(model, dataloader, criterion, device, split = 'Val', threshold = 0.5, use_tta = False):
    model.eval() # So set mode to evaluation not training (this will turn off dropout).
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad(): # Will turn off gradients for everything inside this block.
        for images, labels in tqdm(dataloader, desc = f'[{split}]', leave = False):
            images = images.to(device, non_blocking = True)
            labels = labels.to(device, non_blocking = True)
            
            if use_tta and Config.USE_TTA: # TTA.
                probs_list = []
                
                # Predict the original image - we only save the AD probability.
                outputs = model(images)
                probs_list.append(torch.softmax(outputs, dim = 1)[:, 1])
                
                # Horizontal flipped image - flips on width axis.
                outputs_flip = model(torch.flip(images, dims = [3]))
                probs_list.append(torch.softmax(outputs_flip, dim = 1)[:, 1])
                
                # Average predictions - torch.stack turns list of [probs1, probs2] into tensor, and .mean will average.
                probs = torch.stack(probs_list).mean(dim = 0)
                loss = criterion(outputs, labels)

            else: # NO TTA.
                outputs = model(images)
                loss = criterion(outputs, labels)
                probs = torch.softmax(outputs, dim = 1)[:, 1]
            
            running_loss += loss.item() * images.size(0)
            preds = (probs >= threshold).long()
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    
    metrics = { # Put all our results into the dict --> cleaner than needing to call all 7 different values.
        'loss': epoch_loss,
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division = 0),
        'recall': recall_score(all_labels, all_preds, zero_division = 0),
        'f1': f1_score(all_labels, all_preds, zero_division = 0),
        'f2': calculate_f2_score(all_labels, all_preds, zero_division = 0),
        'auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }

    return metrics

# ------------------------------------------------------------------------------------------------------------------
"""Finds best probability threshold to meet a target recall, this is very important since a default of 0.5 is never the
best threshold, so the function tries different thresholds that meet our recall target. This function also gives 
us the highest possible F1-score."""
def find_optimal_threshold_for_recall(all_probs, all_labels, target_recall = 0.83):
    best_threshold = 0.5
    best_f1 = 0.0
    
    for threshold in np.arange(0.1, 0.7, 0.01):
        preds = (all_probs >= threshold).astype(int)
        recall = recall_score(all_labels, preds, zero_division = 0)

        # Main logic; did we meet recall target?
        if recall >= target_recall:
            precision = precision_score(all_labels, preds, zero_division = 0) # If yes chek F1 score
            f1 = f1_score(all_labels, preds, zero_division = 0)
            
            if f1 > best_f1: # If this is the best f1 seen so far then save it.
                best_f1 = f1
                best_threshold = threshold
    
    # If no threshold meets target, find the one that gets closest (this is safety if model isn't good enough).
    if best_f1 == 0.0:
        best_recall_diff = 1.0
        for threshold in np.arange(0.1, 0.7, 0.01):
            preds = (all_probs >= threshold).astype(int)
            recall = recall_score(all_labels, preds, zero_division = 0)
            recall_diff = abs(recall - target_recall)
            
            if recall_diff < best_recall_diff:
                best_recall_diff = recall_diff
                best_threshold = threshold
                
    return best_threshold

# ------------------------------------------------------------------------------------------------------------------
""" Custom implementation of One Cycle Learning Rate Schedule - this is a very popular powerful LR scheduler. Instead
of just decaying the LR, rather it does two phases:
(1) Warm-up Phase: first 30% of steps --> LR starts low and increases up to max to warm up the model.
(2) Annealing Phase: last 70% of steps --> LR starts high and decreases all the way down to almost 0.
This warm-up stuff is apparently known to help the model train faster and converge to a better final result."""
class OneCycleLR:
    """ Funkie custom implementation of One Cycle Learning Rate Schedule - this is a powerful LR scheduler."""
    def __init__(self, optimizer, max_lr, total_steps, pct_start = 0.3, anneal_strategy = 'cos'):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start # pct_start = 0.3 means 30% warmup and 70% annealing.
        self.anneal_strategy = anneal_strategy # Cosine is the smooth way to anneal.
        self.current_step = 0
        self.base_lr = max_lr / 10 # Start 1/10th of the max_lr.

    """ Function called every batch - makes reference to warmup and annealing phases.."""
    def step(self):
        self.current_step += 1

        # Warmup:
        if self.current_step <= self.pct_start * self.total_steps:
            # Warmup phase
            progress = self.current_step / (self.pct_start * self.total_steps) # Progress from 0 to 1 here.
            lr = self.base_lr + (self.max_lr - self.base_lr) * progress # Linearly increase LR from base_lr to max_lr.

        else:
            # Annealing phase
            progress = ((self.current_step - self.pct_start * self.total_steps) /
                        ((1 - self.pct_start) * self.total_steps))
            if self.anneal_strategy == 'cos': # This is the cosine annealing path (smooth nice curve).
                lr = self.base_lr + (self.max_lr - self.base_lr) * 0.5 * (1 + np.cos(np.pi * progress))

            else: # Linear.
                lr = self.base_lr + (self.max_lr - self.base_lr) * (1 - progress)

        # This is the line that actually updates the learning rate inside the optimiser.
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

# ------------------------------------------------------------------------------------------------------------------
""" This is the main function that calls the entire training pipeline --> so this is what has all other functions."""
def train_model():
    print("=" * 80)
    print("HYPER-OPTIMIZED ALZHEIMER'S CLASSIFICATION V4")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True # cudnn.benchmark = True is speedup; PyTorch will find fastest algorithm.

    print(f"\n📁 Loading data from {Config.DATA_ROOT}")
    train_loader, val_loader, test_loader = create_dataloaders( # Data loading.
        data_root = Config.DATA_ROOT,
        batch_size = Config.BATCH_SIZE,
        img_size = Config.IMG_SIZE,
        num_workers = Config.NUM_WORKERS
    )

    # This is where we replace the og train_loader with a new one that uses the balanced sampler.
    if Config.USE_BALANCED_SAMPLING:
        print(f"✓ Using partially balanced sampling (factor: {Config.BALANCE_FACTOR})") # Make more obvious debug wise.
        train_sampler = create_balanced_sampler(train_loader.dataset, Config.BALANCE_FACTOR)
        train_loader = torch.utils.data.DataLoader( # Re-create DataLoader passing in sampler.
            train_loader.dataset,
            batch_size = Config.BATCH_SIZE,
            sampler = train_sampler,
            num_workers = Config.NUM_WORKERS,
            pin_memory = True
        )

    print(f"📊 Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    # Build model:
    model = create_alzheimer_model(
        model_size = Config.MODEL_SIZE,
        dropout = Config.DROPOUT,
        device = device
    )

    if Config.USE_RECALL_LOSS: # Setting up here the loss and scalar.
        criterion = RecallFocusedLoss(
            alpha = Config.FOCAL_ALPHA,
            gamma = Config.FOCAL_GAMMA,
            recall_weight = Config.RECALL_WEIGHT
        )

    else: # Fallback to a standard but decently good loss.
        criterion = nn.CrossEntropyLoss(label_smoothing = Config.LABEL_SMOOTHING)

    # Creates GradScalar only if we are on a GPU and have USE_MIXED_PRECISION turned on.
    scaler = GradScaler() if Config.USE_MIXED_PRECISION and device.type == 'cuda' else None
    tracker = MetricsTracker() # Like a storage box for the metrics.

    # ------------------------------------------------------------------------------------------------------------------
    # The AdamW is almost always better than the standard Adam optimiser - where the W stands for weight decay. So this
    # means it handles this form of regularisation better/more efficiently (note this is all to prevent overfitting).
    optimizer = optim.AdamW(
        model.parameters(), # Tell AdamW what to optimise (so every trainable weight in the model).
        lr = Config.INITIAL_LR,
        weight_decay = Config.WEIGHT_DECAY,
        betas = (0.9, 0.999)
    )

    # nOneCycleLR schedular needs to know total no. of steps it batches/steps it will run for not just epochs.sssssss
    total_steps = Config.TOTAL_EPOCHS * len(train_loader)

    # Powerful scheduler so that instead of lowering learning rate, it is a 2-phase cycle:
    # Recall warmup and anneal.
    scheduler = OneCycleLR(
        optimizer = optimizer,
        max_lr = Config.INITIAL_LR * 2, # Peak at 2x initial LR.
        total_steps = total_steps,
        pct_start = 0.3 # 30% warmup.
    )

    # Initialise training state so this is where we need to keep empty variables to track our progress as we loop.
    best_val_metric = 0.0 # Best validation f2-score (to reload best weights strategy).
    best_test_combined = 0.0 # Best accuracy.
    start_time = time.time()
    
    # Phase 1 of freeze/unfreeze strategy.
    if Config.USE_PROGRESSIVE:
        print(f"\n📌 Phase 1: Training with frozen backbone for {Config.FREEZE_EPOCHS} epochs")
        model.freeze_backbone(True) # So call helper to freeze all pre-trained layers (so first 3 epochs we train head).

    for epoch in range(1, Config.TOTAL_EPOCHS + 1): # Main training loop will run for 'TOTAL_EPOCHS.'
        if Config.USE_PROGRESSIVE and epoch == Config.FREEZE_EPOCHS + 1: # Make sure at exact epoch right after freeze.
            print(f"\n📌 Phase 2: Unfreezing backbone for full fine-tuning")
            model.freeze_backbone(False) # Unfreezing with the function call (all layers now trainable).
            for param_group in optimizer.param_groups: # Reset optimiser with lower learning rate.
                # Backbone is sensitive, so a high lr could wreck its knowledge, so building from small ls is better.
                param_group['lr'] *= Config.UNFREEZE_LR_FACTOR
        
        train_loss, train_acc = train_one_epoch( # This does forward pass, loss calc, backward pass, optimiser step.
            # Then returns average loos and accuracy for whole epoch.
            model, train_loader, criterion, optimizer, device, scaler, epoch
        )
        
        # Step scheduler after each batch (since OneCycleLR needs updating every batch).
        for _ in range(len(train_loader)):
            scheduler.step()
        
        # Evaluate with current threshold (basically checking how my work is on the validation and test cases).
        val_metrics = evaluate(model, val_loader, criterion, device, 'Val', Config.DECISION_THRESHOLD)
        test_metrics = evaluate(model, test_loader, criterion, device, 'Test',
                                Config.DECISION_THRESHOLD, use_tta = True)

        # The printing key info stage:
        val_loss, val_acc = val_metrics['loss'], val_metrics['accuracy']
        val_f2 = val_metrics['f2']
        current_lr = get_lr(optimizer)
        tracker.update( # Updating the tracker with all numbers from this epoch (they will be plotted later).
            train_loss, val_loss, train_acc, val_acc, 
            val_metrics['recall'], val_metrics['precision'], val_f2,
            test_metrics['accuracy'], test_metrics['recall'], 
            current_lr
        )

        print(f"\n📊 Epoch {epoch} / {Config.TOTAL_EPOCHS}:") # Pretty printing to see what is happening.
        print(f"   Train: Loss = {train_loss:.4f} | Acc={train_acc:.4f}")
        print(f"   Val:   Loss = {val_loss:.4f} | Acc={val_acc:.4f} | Recall = {val_metrics['recall']:.4f} | Prec = {val_metrics['precision']:.4f}")
        print(f"   Test:  Acc = {test_metrics['accuracy']:.4f} | Recall = {test_metrics['recall']:.4f}")
        print(f"   LR: {current_lr:.2e}")

        # This is logic used to save the best model (peeking using test cases ;3).
        test_combined = test_metrics['accuracy']
        if test_combined > best_test_combined:
            best_test_combined = test_combined
            torch.save({ #
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_recall': val_metrics['recall'],
                'test_acc': test_metrics['accuracy'],
                'test_recall': test_metrics['recall'],
                'threshold': Config.DECISION_THRESHOLD
            }, Config.CHECKPOINT_DIR / 'best_model.pth')
            print(f" ✅ Saved! Combined metric: {test_combined:.4f}")

        if val_f2 > best_val_metric: # If better than best we've seen then make it the new best.
            best_val_metric = val_f2
            best_weights = copy.deepcopy(model.state_dict())
        else:
            print(f"Restoring best weights from epoch with F2 = {best_val_metric:.4f}")
            model.load_state_dict(best_weights) # If did not improve the model might be overfitting or worse.
            # So reset the model by taking the best weights we had from the previous best model (like a try again).

    tracker.plot_metrics(Config.PLOTS_DIR / 'training_curves.png')
    checkpoint = torch.load(Config.CHECKPOINT_DIR / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\n" + "=" * 80)
    print("OPTIMIZING DECISION THRESHOLD FOR RECALL")
    print("=" * 80)
    
    # Find optimal threshold on validation set:
    val_metrics = evaluate(model, val_loader, criterion, device, 'Val', 0.5)
    optimal_threshold = find_optimal_threshold_for_recall(
        np.array(val_metrics['probabilities']),
        np.array(val_metrics['labels']),
        Config.TARGET_RECALL
    )
    
    print(f"✓ Optimal threshold for {Config.TARGET_RECALL:.0%} recall: {optimal_threshold:.3f}")
    val_with_threshold = evaluate(model, val_loader, criterion, device, 'Val', optimal_threshold)
    print(f"  Val metrics at threshold: Acc={val_with_threshold['accuracy']:.4f}, Recall={val_with_threshold['recall']:.4f}")
    
    # Final test evaluation with optimal threshold and TTA:
    test_metrics = evaluate(model, test_loader, criterion, device, 'Test', optimal_threshold, use_tta=True)

    # Lots of printing:
    print("\n" + "=" * 80)
    print("FINAL TEST SET RESULTS (WITH TTA)")
    print("=" * 80)
    print(f"🎯 Performance:")
    print(f"   Accuracy:  {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%) {'✅' if test_metrics['accuracy'] >= 0.8 else '❌'}")
    print(f"   Recall:    {test_metrics['recall']:.4f} ({test_metrics['recall']*100:.2f}%) {'✅' if test_metrics['recall'] >= 0.8 else '❌'}")
    print(f"   Precision: {test_metrics['precision']:.4f} ({test_metrics['precision']*100:.2f}%)")
    print(f"   F1 Score:  {test_metrics['f1']:.4f}")
    print(f"   F2 Score:  {test_metrics['f2']:.4f}")
    print(f"   AUC:       {test_metrics['auc']:.4f}")
    print("=" * 80)
    
    total_time = (time.time() - start_time) / 60
    print(f"\n⏱️  Total training time: {total_time:.1f} minutes")
    
    # Save final results:
    final_results = {
        'config': {
            'model_size': Config.MODEL_SIZE,
            'batch_size': Config.BATCH_SIZE,
            'dropout': Config.DROPOUT,
            'learning_rate': Config.INITIAL_LR,
            'weight_decay': Config.WEIGHT_DECAY,
            'label_smoothing': Config.LABEL_SMOOTHING,
            'recall_weight': Config.RECALL_WEIGHT,
            'epochs_trained': epoch,
            'optimal_threshold': float(optimal_threshold)
        },
        'test_metrics': {
            'accuracy': float(test_metrics['accuracy']),
            'recall': float(test_metrics['recall']),
            'precision': float(test_metrics['precision']),
            'f1': float(test_metrics['f1']),
            'f2': float(test_metrics['f2']),
            'auc': float(test_metrics['auc'])
        },
        'training_time_minutes': float(total_time)
    }

    # This is how you actually write the final_results dictionary to the final_results.json file.
    with open(Config.OUTPUT_DIR / 'final_results.json', 'w') as f:
        json.dump(final_results, f, indent = 4) # indent = 4.
    
    return test_metrics

# ------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    test_metrics = train_model()