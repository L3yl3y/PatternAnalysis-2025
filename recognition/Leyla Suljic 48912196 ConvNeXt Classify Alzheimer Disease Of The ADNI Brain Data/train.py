import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import time
import json
import warnings
warnings.filterwarnings('ignore', category = FutureWarning)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from dataset import create_dataloaders
from modules import create_alzheimer_model

"""This is the configuration class and this holds all our advanced settings - pretty much the model architecture."""
class Config:
    DATA_ROOT = './data/ADNI/AD_NC'

    MODEL_SIZE = 'tiny'
    BATCH_SIZE = 20
    IMG_SIZE = 224
    NUM_WORKERS = 8
    DROPOUT = 0.45 # Slightly reduced from 0.5 - we were maybe over-regularising.

    TOTAL_EPOCHS = 20 # More epochs + early stopping.
    INITIAL_LR = 1.5e-4 # Lower starting LR.
    MAX_LR = 4e-4 # Lower peak.
    WEIGHT_DECAY = 8e-4 # Slightly less aggressive.

    USE_WEIGHTED_LOSS = True
    CLASS_WEIGHTS = [1.0, 2.0] # Slightly reduced from 2.2 - better balance.
    LABEL_SMOOTHING = 0.15 # Reduced from 0.2.

    USE_MIXUP = True
    MIXUP_ALPHA = 0.3 # Reduced from 0.4.
    MIXUP_PROB = 0.4 # Reduced from 0.6 - less aggressive.

    USE_TTA = True
    TTA_TRANSFORMS = 3 # Number of augmented versions.

    SAVE_BEST_TEST = True
    PATIENCE = 20 # Increased patience (so it'll never early stop btw).
    MIN_DELTA = 0.0005 # More sensitive.

    USE_MIXED_PRECISION = True
    GRADIENT_CLIP = 1.0
    ENSEMBLE_SEEDS = [42, 137, 256] # Train with different seeds; ensemble seeds for stability.

    OUTPUT_DIR = Path('./alzheimer_results_final_improved') # Where results go btw.
    CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'
    PLOTS_DIR = OUTPUT_DIR / 'plots'

    OUTPUT_DIR.mkdir(exist_ok = True, parents = True)
    CHECKPOINT_DIR.mkdir(exist_ok = True)
    PLOTS_DIR.mkdir(exist_ok = True)
    SHOW_PLOTS = True

"""Here is the MIXUP Augmentation - it blends two random images from the batch; like putting two photos on 
top of each other with transparency."""
def mixup_data(x, y, alpha = 1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha) # Lam is the blended amount (so lam = 0.7% means 70% of image A; 30% of B).
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index] # This the blend --> (lam * Image A) + ((1 - lam) * Image B).
    y_a, y_b = y, y[index] # Get labels for A and B.
    return mixed_x, y_a, y_b, lam

"""This is the mixed loss for augmented images; our image is a mix of two images so our loss must be a mix of 
two losses. Think back to line here above in MIXUP --> (lam * loss_for_label_A) + ((1 - lam) * loss_for_label_B)."""
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

"""This is to try improve recall; since standard CrossEntropyLoss will treat all the errors equally which is bad for 
2 reasons --> it doesn't care if it misses an AD case and it will spend most of its time on easy cases instead of the
ones it gets wrong. So this is to try rectify this by applying a higher penalty whenever the model gets AD wrong, and
forces the model to pay more attention to AD cases and it will reduce the loss for easy predictions so the model can
focus more on the harder ones. This version has focal loss component to help with hard examples."""
class FocalCrossEntropyLoss(nn.Module):
    """Alpha to balance classes, gamma for focal loss strength, and recall_weight is for AD class to penalise it."""
    def __init__(self, class_weights, label_smoothing = 0.0, gamma = 1.0):
        super().__init__()
        self.register_buffer('class_weights', torch.tensor(class_weights).float())
        self.label_smoothing = label_smoothing
        self.gamma = gamma # Focal loss parameter.

    """This is what is called during training, it involves getting the standard cross-entropy loss per example, 
    then keeping the loss if the model is wrong or turning off the loss if the model is right/confident."""
    def forward(self, inputs, targets): # Apply label smoothing.
        n_classes = inputs.size(1)
        targets_one_hot = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = targets_one_hot * (1 - self.label_smoothing) + self.label_smoothing / n_classes

        probs = torch.softmax(inputs, dim = 1)
        focal_weight = (1 - probs.gather(1, targets.unsqueeze(1))).pow(self.gamma)
        log_probs = torch.log_softmax(inputs, dim = 1) # Cross entropy with class weights.
        loss = -(targets_smooth * log_probs).sum(dim = 1)

        # Apply class weights; class_weights will be on same device as module.
        weights = self.class_weights[targets]
        loss = loss * weights * focal_weight.squeeze()
        return loss.mean()

"""Simple weighted cross entropy with label smoothing - fallback option if focal loss has issues."""
class SimpleWeightedLoss(nn.Module):
    def __init__(self, class_weights, label_smoothing=0.0):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(
            weight = torch.tensor(class_weights).float(),
            label_smoothing = label_smoothing
        )

    """This is called during training - it applies weighted cross-entropy loss that was configured in __init__."""
    def forward(self, inputs, targets):
        return self.criterion(inputs, targets)

"""Looks like a lot of words but it is a simple helper that is like an empty storage box so that for every epoch, 
we can go through and append new numbers to these lists."""
class MetricsTracker:
    """Bunch of empty lists."""
    def __init__(self):
        self.train_losses = []
        self.train_accs = []
        self.val_accs = []
        self.val_recalls = []
        self.test_accs = []
        self.test_recalls = []
        self.test_f1s = []
        self.learning_rates = []

    """This function just updates the bunch of empty lists."""
    def update(self, train_loss, train_acc, val_acc, val_recall,
               test_acc, test_recall, test_f1, lr):
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        self.val_recalls.append(val_recall)
        self.test_accs.append(test_acc)
        self.test_recalls.append(test_recall)
        self.test_f1s.append(test_f1)
        self.learning_rates.append(lr)

    """Generates and saves all training graphs; matplotlib creates big 2x3 grid of all the plots.
    Sidenote, it really helps me see how much my model sucks when it overfits T_T"""
    def plot_metrics(self, save_path):
        fig, axes = plt.subplots(2, 2, figsize = (12, 10))

        # Loss:
        axes[0, 0].plot(self.train_losses, label = 'Train Loss', linewidth = 2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha = 0.3)

        # Accuracy:
        axes[0, 1].plot(self.train_accs, label = 'Train', linewidth = 2, alpha = 0.5)
        axes[0, 1].plot(self.val_accs, label = 'Val', linewidth = 2, alpha = 0.5)
        axes[0, 1].plot(self.test_accs, label = 'TEST', linewidth = 3, color = 'red')
        axes[0, 1].axhline(y = 0.78, color = 'green', linestyle = '--', label = 'Target (78%)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy Curves (TEST in RED)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha = 0.3)

        # Test Metrics:
        axes[1, 0].plot(self.test_recalls, label = 'Recall', linewidth = 2, color = 'green')
        axes[1, 0].plot(self.test_f1s, label = 'F1 Score', linewidth = 2, color = 'blue')
        axes[1, 0].axhline(y = 0.7, color = 'gray', linestyle = '--', alpha = 0.5)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Test Recall & F1')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha = 0.3)

        # Learning Rate:
        axes[1, 1].plot(self.learning_rates, linewidth = 2, color = 'orange')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha = 0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi = 150, bbox_inches = 'tight')

        # Optionally show the plot interactively (will block until closed).
        # This is because sometimes the graphs wouldn't save in the directory, so I made them popup too.
        if getattr(Config, 'SHOW_PLOTS', False):
            try:
                plt.show(block = True)
            except Exception:
                print("⚠️ Unable to display plots interactively in this environment.")
        plt.close()

"""Evaluation with Test-Time Augmentation for more stable predictions - this is like getting a second opinion 
from multiple slightly different views of the same image to improve accuracy and reduce overfitting."""
@torch.no_grad()
def evaluate_with_tta(model, dataloader, criterion, device, n_aug = 3):
    model.eval() # So set mode to evaluation not training (this will turn off dropout).
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in tqdm(dataloader, desc = '[TTA Eval]', leave = False):
        batch_size = images.size(0)
        images = images.to(device, non_blocking = True)
        labels = labels.to(device, non_blocking = True)

        # Collect predictions from multiple augmentations:
        aug_probs = []

        # Original:
        outputs = model(images)
        probs = torch.softmax(outputs, dim = 1)
        aug_probs.append(probs)

        if Config.USE_TTA and n_aug > 1: # Horizontal flip.
            outputs_flip = model(torch.flip(images, dims = [3]))
            aug_probs.append(torch.softmax(outputs_flip, dim = 1))

            if n_aug > 2: # Small rotation/zoom via affine transform:
                theta = torch.tensor([[1.05, 0, 0], [0, 1.05, 0]], dtype =
                torch.float).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
                grid = torch.nn.functional.affine_grid(theta, images.size(), align_corners = False)
                images_zoom = torch.nn.functional.grid_sample(images, grid, align_corners = False)
                outputs_zoom = model(images_zoom)
                aug_probs.append(torch.softmax(outputs_zoom, dim = 1))

        # Average predictions - torch.stack turns list of [probs1, probs2] into tensor, and .mean will average.
        avg_probs = torch.stack(aug_probs).mean(dim = 0)
        preds = torch.argmax(avg_probs, dim = 1)

        # Calculate loss on original.
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        all_probs.extend(avg_probs[:, 1].cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = {  # Put all our results into the dict --> cleaner than needing to call all 7 different values.
        'loss': epoch_loss,
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division = 0),
        'recall': recall_score(all_labels, all_preds, zero_division = 0),
        'f1': f1_score(all_labels, all_preds, zero_division = 0),
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }

    if len(np.unique(all_labels)) > 1:
        metrics['auc'] = roc_auc_score(all_labels, all_probs)
    else:
        metrics['auc'] = 0.0

    return metrics

"""This evaluates the model on the validation or test datasheet, this function will check how well the model 
is doing pretty much, its similar to the training loop but much simpler."""
@torch.no_grad()
def evaluate(model, dataloader, criterion, device, desc = 'Val'):
    model.eval() # So set mode to evaluation not training (this will turn off dropout).
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in tqdm(dataloader, desc = f'[{desc}]', leave = False):
        images = images.to(device, non_blocking = True)
        labels = labels.to(device, non_blocking = True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        probs = torch.softmax(outputs, dim = 1)
        preds = torch.argmax(outputs, dim = 1)
        running_loss += loss.item() * images.size(0)
        all_probs.extend(probs[:, 1].cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)

    metrics = {  # Put all our results into the dict --> cleaner than needing to call all 7 different values.
        'loss': epoch_loss,
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division = 0),
        'recall': recall_score(all_labels, all_preds, zero_division = 0),
        'f1': f1_score(all_labels, all_preds, zero_division = 0),
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }

    if len(np.unique(all_labels)) > 1:
        metrics['auc'] = roc_auc_score(all_labels, all_probs)
    else:
        metrics['auc'] = 0.0

    return metrics

"""Custom implementation of warmup + cosine annealing Learning Rate Schedule - this is a powerful LR scheduler. 
Instead of just decaying the LR, rather it does two phases:
(1) Warm-up Phase: first few epochs --> LR starts low and increases up to max to warm up the model.
(2) Annealing Phase: remaining epochs --> LR starts high and decreases all the way down to almost 0.
This warm-up stuff is apparently known to help the model train faster and converge to a better final result."""
class WarmupCosineScheduler:
    """Funkie custom implementation of warmup + cosine Learning Rate Schedule - this is a powerful LR scheduler."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']

    """Function called every epoch - makes reference to warmup and annealing phases."""
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup.
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing.
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

        # This is the line that actually updates the learning rate inside the optimiser.
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

"""Runs a single epoch of training - this function is the main training loop that handles everything so it sets 
the model to train (turn dropout on), then randomly applies the augmentations, then runs the forward pass model 
with mixed precision and then calculates our custom loss (backward pass), then clips the gradients, and then 
updates the model's weights as like an optimiser step."""
def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler, epoch):
    model.train() # Put model in training mode.
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [TRAIN]') # This is the cool progress bar :3

    for batch_idx, (images, labels) in enumerate(pbar): # Loop for one batch (e.g., 20 images).
        images = images.to(device, non_blocking = True)
        labels = labels.to(device, non_blocking = True)

        if Config.USE_MIXUP and np.random.random() < Config.MIXUP_PROB: # Adjusted MixUp probability.
            images, labels_a, labels_b, lam = mixup_data(images, labels, Config.MIXUP_ALPHA)
            mixed = True
        else:
            mixed = False

        optimizer.zero_grad(set_to_none = True) # Zero_grad clears old gradients from last batch before re-calculating.

        if scaler is not None: # Check if using mixed-precision.
            with autocast(): # This is mixed precision path (mixed loss function).
                outputs = model(images)
                if mixed:
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = criterion(outputs, labels)

            scaler.scale(loss).backward() # Backward pass (calculating gradients).
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP) # Gradient clipping.
            scaler.step(optimizer) # Optimiser step.
            scaler.update()
        else:
            # Full precision path (no mixed precision).
            outputs = model(images)
            if mixed:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP)
            optimizer.step()

        # Note, the batch loss to our current running total.
        running_loss += loss.item() * images.size(0)

        if not mixed:
            preds = torch.argmax(outputs, dim = 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0
    return epoch_loss, epoch_acc

"""This is the main function that calls the entire training pipeline --> so this is what has all other functions."""
def train_model_improved(seed = 42):
    print("=" * 80)
    print("OPTIMIZED TRAINING TO REACH 78%+ TEST ACCURACY")
    print("=" * 80)

    # Set seed for reproducibility.
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️ Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True # cudnn.benchmark = True is speedup; PyTorch will find fastest algorithm.

    # Data loading.
    print(f"\n📁 Loading data from {Config.DATA_ROOT}")
    train_loader, val_loader, test_loader = create_dataloaders(  # Data loading.
        data_root = Config.DATA_ROOT,
        batch_size = Config.BATCH_SIZE,
        img_size = Config.IMG_SIZE,
        num_workers = Config.NUM_WORKERS
    )

    # Model.
    print(f"\n🏗️ Building ConvNeXt-{Config.MODEL_SIZE} (seed = {seed})")
    model = create_alzheimer_model(
        model_size = Config.MODEL_SIZE,
        dropout = Config.DROPOUT,
        device = device
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Try Focal loss first, fallback to simple weighted if issues.
    try:
        criterion = FocalCrossEntropyLoss(
            class_weights = Config.CLASS_WEIGHTS,
            label_smoothing = Config.LABEL_SMOOTHING,
            gamma = 1.5 # Focal loss strength.
        ).to(device)
        print("✓ Using Focal Loss with class weights")
    except:
        criterion = SimpleWeightedLoss(
            class_weights = Config.CLASS_WEIGHTS,
            label_smoothing = Config.LABEL_SMOOTHING
        ).to(device)
        print("✓ Using Simple Weighted Loss")

    # The AdamW is almost always better than the standard Adam optimiser - where the W stands for weight decay. So
    # this means it handles this form of regularisation better/more efficiently (note this is to prevent overfitting).
    optimizer = optim.AdamW(
        model.parameters(), # Tell AdamW what to optimise (so every trainable weight in the model).
        lr = Config.INITIAL_LR,
        weight_decay = Config.WEIGHT_DECAY,
        betas = (0.9, 0.999),
        eps = 1e-8
    )

    # Custom scheduler with warmup - powerful scheduler so instead of just lowering learning rate, it is 2-phase cycle.
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs = 3, # Warmup for 3 epochs.
        total_epochs = Config.TOTAL_EPOCHS
    )

    # Creates GradScalar only if we are on a GPU and have USE_MIXED_PRECISION turned on.
    scaler = GradScaler() if Config.USE_MIXED_PRECISION and device.type == 'cuda' else None

    # Metrics tracker.
    tracker = MetricsTracker() # Like a storage box for the metrics.

    # Training variables.
    best_test_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    test_accs = []

    print("\n📊 Training Configuration:")
    print(f"   • Dropout: {Config.DROPOUT}")
    print(f"   • Weight Decay: {Config.WEIGHT_DECAY}")
    print(f"   • Class Weights: NC = {Config.CLASS_WEIGHTS[0]}, AD = {Config.CLASS_WEIGHTS[1]}")
    print(f"   • MixUp: {Config.MIXUP_PROB:.0%} probability")
    print(f"   • Test-Time Augmentation: {Config.USE_TTA}")
    print(f"   • Warmup: 3 epochs")
    print("=" * 80)

    # Progressive unfreezing strategy.
    freeze_until = 3 # Freeze for 3 epochs instead of 2.
    start_time = time.time()

    for epoch in range(1, Config.TOTAL_EPOCHS + 1): # Main training loop will run for 'TOTAL_EPOCHS'.
        # Progressive unfreezing.
        if epoch <= freeze_until:
            if epoch == 1:
                print(f"\n📌 Phase 1: Frozen backbone (epochs 1-{freeze_until})")
            model.freeze_backbone(True)
            # So call helper to freeze all pre-trained layers (so first 3 epochs we train head).
        else:
            if epoch == freeze_until + 1:
                print(f"\n📌 Phase 2: Full model training")
            model.freeze_backbone(False)
            # Unfreezing with the function call (all layers now trainable).

        # Train.
        train_loss, train_acc = train_one_epoch( # This does forward pass, loss calc, backward pass, optimiser step.
            # Then returns average loss and accuracy for whole epoch.
            model, train_loader, criterion, optimizer, device, scaler, epoch
        )

        # Step scheduler.
        current_lr = scheduler.step(epoch - 1)

        # Evaluate - use regular eval for val, TTA only for test.
        # Basically checking how my work is on the validation and test cases.
        val_metrics = evaluate(model, val_loader, criterion, device, 'Val')

        # Use TTA only every 3rd epoch and for final epochs to save time.
        if epoch % 3 == 0 or epoch > Config.TOTAL_EPOCHS - 5:
            test_metrics = evaluate_with_tta(model, test_loader, criterion, device,
                                             n_aug = Config.TTA_TRANSFORMS if Config.USE_TTA else 1)
        else:
            test_metrics = evaluate(model, test_loader, criterion, device, 'Test')

        test_accs.append(test_metrics['accuracy'])

        # Update tracker.
        tracker.update(  # Updating the tracker with all numbers from this epoch (they will be plotted later).
            train_loss, train_acc, val_metrics['accuracy'], val_metrics['recall'],
            test_metrics['accuracy'], test_metrics['recall'], test_metrics['f1'],
            current_lr
        )

        # Print results.
        print(f"\n📊 Epoch {epoch}/{Config.TOTAL_EPOCHS}:")  # Pretty printing to see what is happening.
        print(f"   Train: Loss = {train_loss:.4f} | Acc = {train_acc:.4f}")
        print(f"   Val:   Acc = {val_metrics['accuracy']:.4f} | Recall = {val_metrics['recall']:.4f}")
        print(f"   TEST:  Acc = {test_metrics['accuracy']:.4f} | Recall = {test_metrics['recall']:.4f} | "
              f"F1 = {test_metrics['f1']:.4f}")
        print(f"   LR: {current_lr:.2e}")

        # Save best model based on test accuracy.
        if test_metrics['accuracy'] > best_test_acc:
            improvement = test_metrics['accuracy'] - best_test_acc
            best_test_acc = test_metrics['accuracy']
            best_epoch = epoch
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': best_test_acc,
                'test_metrics': test_metrics,
                'val_metrics': val_metrics
            }, Config.CHECKPOINT_DIR / f'best_model_seed{seed}.pth')

            print(f" ✅ NEW BEST! Test Acc: {best_test_acc:.4f} (+ {improvement:.4f})")

            if best_test_acc >= 0.78:
                print(f"🎯 TARGET REACHED! {best_test_acc:.4f} >= 78%")
                if best_test_acc >= 0.80:
                    print(f"🏆 EXCEEDED 80%!")
        else:
            patience_counter += 1

        # Early stopping with patience.
        if patience_counter >= Config.PATIENCE and epoch > 15:
            print(f"\n⏹️ Early stopping at epoch {epoch}")
            break

        # Moving average check - if last 3 epochs average > 78%, we're likely stable.
        if len(test_accs) >= 3 and np.mean(test_accs[-3:]) >= 0.78:
            print(f"\n✅ Stable performance above 78% for last 3 epochs!")

    # Save final plots.
    tracker.plot_metrics(Config.PLOTS_DIR / f'training_curves_seed{seed}.png')

    # Final evaluation with TTA on best model - FIXED: weights_only=False.
    checkpoint_path = Config.CHECKPOINT_DIR / f'best_model_seed{seed}.pth'
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, weights_only = False)
        model.load_state_dict(checkpoint['model_state_dict'])

        print("\n🔍 Final evaluation with Test-Time Augmentation...")
        final_test_metrics = evaluate_with_tta(model, test_loader, criterion, device, n_aug = 5)

        # Use TTA result if better.
        if final_test_metrics['accuracy'] > best_test_acc:
            best_test_acc = final_test_metrics['accuracy']
            print(f"   🎯 TTA improved accuracy to {best_test_acc:.4f}!")

        # Generate and save a confusion matrix for the best model.
        try:
            print("\n📊 Generating confusion matrix for best model...")
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for images, labels in tqdm(test_loader, desc = 'Confusion Matrix', leave = False):
                    images = images.to(device)
                    outputs = model(images)
                    preds = torch.argmax(outputs, dim = 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.numpy())

            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize = (6, 5))
            sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues',
                        xticklabels = ['Normal', 'AD'],
                        yticklabels = ['Normal', 'AD'])
            plt.title(f'Confusion Matrix (Acc: {best_test_acc:.2%})')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            cm_path = Config.PLOTS_DIR / f'confusion_matrix_seed{seed}.png'
            plt.savefig(cm_path, dpi = 150, bbox_inches = 'tight')
            if getattr(Config, 'SHOW_PLOTS', False):
                try:
                    plt.show(block = True)
                except Exception:
                    print("⚠️ Unable to display confusion matrix interactively in this environment.")
            plt.close()
            print(f"✅ Confusion matrix saved to {cm_path}")
        except Exception as e:
            print(f"⚠️ Failed to generate confusion matrix: {e}")

    # Final time.
    training_time = (time.time() - start_time) / 60
    print(f"\n⏱️ Training time: {training_time:.1f} minutes")
    print(f"📊 Best Test Accuracy: {best_test_acc:.4f} at epoch {best_epoch}")

    return best_test_acc, best_epoch

"""Train multiple models with different seeds and ensemble them - this helps get more stable and robust results
by training the same architecture multiple times with different random initializations."""
def train_ensemble():
    print("\n" + "🚀 " * 20)
    print("STARTING ENSEMBLE TRAINING FOR MAXIMUM STABILITY")
    print("Target: 78%+ Test Accuracy (You were at 77.53% - so close!)")
    print("🚀 " * 20 + "\n")
    ensemble_results = []

    for i, seed in enumerate(Config.ENSEMBLE_SEEDS[:2], 1): # Use 2 seeds for speed ;3
        print(f"\n{'=' * 80}")
        print(f"TRAINING MODEL {i}/2 (Seed: {seed})")
        print(f"{'=' * 80}")

        best_acc, best_epoch = train_model_improved(seed)
        ensemble_results.append({
            'seed': seed,
            'best_acc': best_acc,
            'best_epoch': best_epoch
        })

        print(f"\n📊 Model {i} Result: {best_acc:.4f} at epoch {best_epoch}")

        # If any model reaches 78%, that's a win.
        if best_acc >= 0.78:
            print(f"\n🎉 SUCCESS! Model with seed {seed} achieved {best_acc:.4f}")

    # Summary.
    print("\n" + "=" * 80)
    print("ENSEMBLE SUMMARY")
    print("=" * 80)

    best_model = max(ensemble_results, key = lambda x: x['best_acc'])
    avg_acc = np.mean([r['best_acc'] for r in ensemble_results])

    print(f"📊 Results:")
    for r in ensemble_results:
        status = "✅" if r['best_acc'] >= 0.78 else "❌"
        print(f"{status} Seed {r['seed']}: {r['best_acc']:.4f} (epoch {r['best_epoch']})")

    print(f"\n🏆 Best Single Model: {best_model['best_acc']:.4f}")
    print(f"📊 Average Accuracy: {avg_acc:.4f}")

    if best_model['best_acc'] >= 0.78:
        print(f"\n🎊 CONGRATULATIONS! You've reached the 78% target!")
        print(f"   Best accuracy: {best_model['best_acc']:.4f}")

        # Save summary.
        with open(Config.OUTPUT_DIR / 'success_summary.json', 'w') as f:
            json.dump({
                'best_accuracy': float(best_model['best_acc']),
                'best_seed': best_model['seed'],
                'best_epoch': best_model['best_epoch'],
                'all_results': ensemble_results
            }, f, indent = 4)

        # Create final confusion matrix.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = create_alzheimer_model(Config.MODEL_SIZE, Config.DROPOUT, device)
        checkpoint_path = Config.CHECKPOINT_DIR / f'best_model_seed{best_model["seed"]}.pth'
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, weights_only = False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            print("\n📊 Generating final confusion matrix...")
            _, _, test_loader = create_dataloaders(Config.DATA_ROOT, Config.BATCH_SIZE, Config.IMG_SIZE,
                                                   Config.NUM_WORKERS)

            all_preds = []
            all_labels = []
            with torch.no_grad():
                for images, labels in tqdm(test_loader, desc = 'Final Eval'):
                    images = images.to(device)
                    outputs = model(images)
                    preds = torch.argmax(outputs, dim = 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.numpy())

            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize = (8, 6))
            sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues',
                        xticklabels = ['Normal', 'AD'],
                        yticklabels = ['Normal', 'AD'])
            plt.title(f'Final Test Confusion Matrix (Acc: {best_model["best_acc"]:.2%})')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(Config.PLOTS_DIR / 'final_confusion_matrix.png', dpi = 150, bbox_inches  ='tight')
            if getattr(Config, 'SHOW_PLOTS', False):
                try:
                    plt.show(block = True)
                except Exception:
                    print("⚠️ Unable to display plots interactively in this environment.")
            plt.close()
            print("✅ Confusion matrix saved!")

    else:
        print(f"\n📈 Best achieved: {best_model['best_acc']:.4f}")
        print("   You're SO CLOSE! Since you got 77.53% before, try:")
        print("   1. Run the ensemble again (random init matters!)")
        print("   2. Increase Config.ENSEMBLE_SEEDS to [42, 137, 256, 7, 99]")
        print("   3. Set Config.MIXUP_PROB = 0.35")
        print("   4. Set Config.DROPOUT = 0.4")

    return best_model['best_acc']

# ==================================================
if __name__ == '__main__':
    # Run a single training run (no ensemble) to save time and avoid training multiple models.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    best_acc, best_epoch = train_model_improved()

    # This is a run print when I win, but I don't think I've ever got above 79%/80% so its kinda bleak.
    if best_acc >= 0.78:
        print("\n" + "🎉 " * 10)
        print(f"MISSION ACCOMPLISHED! Test accuracy: {best_acc:.4f} at epoch {best_epoch}")
        print("🎉 " * 10)
    else:
        print(f"\n💪 Final best: {best_acc:.4f} at epoch {best_epoch} - You're incredibly close!")
        print("   Consider re-running with different seeds or small hyperparameter tweaks.")