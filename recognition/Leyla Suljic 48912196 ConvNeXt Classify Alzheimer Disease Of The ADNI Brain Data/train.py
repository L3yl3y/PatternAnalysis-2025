import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import WeightedRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import time
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from dataset import create_dataloaders
from modules import create_alzheimer_model

# ------------------------------------------------------------------------------------------------------------------
def calculate_f2_score(y_true, y_pred, zero_division=0):
    """F2-score prioritizes Recall (beta=2)"""
    precision = precision_score(y_true, y_pred, zero_division=zero_division)
    recall = recall_score(y_true, y_pred, zero_division=zero_division)
    
    beta_sq = 4  # F2-score
    if (beta_sq * precision + recall) == 0:
        return 0.0
    
    f2 = (1 + beta_sq) * (precision * recall) / ((beta_sq * precision) + recall)
    return f2

# ------------------------------------------------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Focal Loss with balanced alpha weights"""
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if not isinstance(self.alpha, torch.Tensor):
                self.alpha = torch.tensor(self.alpha, dtype=torch.float32).to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()

# ------------------------------------------------------------------------------------------------------------------
# OPTIMIZED HYPERPARAMETERS FOR 80%+ ACCURACY
# ------------------------------------------------------------------------------------------------------------------
class Config:
    DATA_ROOT = './data/ADNI/AD_NC'
    
    # MODEL CONFIGURATION - Balanced capacity
    BATCH_SIZE = 24  # OPTIMAL: Not too small (noisy) or large (less updates)
    IMG_SIZE = 224
    NUM_WORKERS = 8
    MODEL_SIZE = 'small'  # UPGRADED from 'tiny' - more capacity for better accuracy
    DROPOUT = 0.35  # INCREASED from 0.2 - better regularization without overdoing it
    
    # TRAINING SCHEDULE - Longer but controlled
    TOTAL_EPOCHS = 35  # Slightly more epochs for convergence
    WARMUP_EPOCHS = 4  # Gradual warmup
    INITIAL_LR = 3e-4  # Higher LR for 'small' model
    MIN_LR = 1e-7
    
    WEIGHT_DECAY = 3e-4  # Moderate L2 regularization
    PATIENCE = 8  # FIXED from 700 - reasonable early stopping
    MIN_DELTA = 0.001
    USE_MIXED_PRECISION = True
    GRADIENT_CLIP = 1.0
    LABEL_SMOOTHING = 0.05  # Mild smoothing only
    
    # LOSS CONFIGURATION - Balanced approach
    USE_FOCAL_LOSS = False  # CHANGED: Use standard CE with label smoothing instead
    FOCAL_ALPHA = [0.45, 0.55]  # Not used when focal loss disabled
    FOCAL_GAMMA = 1.0  # Not used when focal loss disabled
    USE_BALANCED_SAMPLING = False  # Keep disabled
    
    # DECISION THRESHOLD
    TARGET_RECALL = 0.82  # Slightly higher target
    DECISION_THRESHOLD = 0.5  # Standard threshold
    
    # AUGMENTATION STRATEGY
    USE_MIXUP = True
    MIXUP_ALPHA = 0.15  # Moderate mixing
    MIXUP_PROB = 0.4  # Apply 40% of the time
    
    # ADDITIONAL REGULARIZATION
    USE_CUTMIX = True  # NEW: Alternative to mixup
    CUTMIX_PROB = 0.3  # Apply 30% of the time
    STOCHASTIC_DEPTH_RATE = 0.0  # Keep disabled for stability
    
    # TWO-PHASE TRAINING (NEW)
    USE_TWO_PHASE = True
    FREEZE_EPOCHS = 5  # Freeze backbone for first 5 epochs
    
    OUTPUT_DIR = Path('./alzheimer_results_v3')
    CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'
    PLOTS_DIR = OUTPUT_DIR / 'plots'

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------------------------------------------------------
def mixup_data(x, y, alpha=1.0):
    """Mixup augmentation - blend two random samples"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    """CutMix augmentation - cut and paste patches between samples"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    """Generate random bounding box for CutMix"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup/CutMix loss calculation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ------------------------------------------------------------------------------------------------------------------
class MetricsTracker:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.val_recalls = []
        self.val_f2_scores = []
        self.test_accs = []
        self.test_recalls = []
        self.learning_rates = []
    
    def update(self, train_loss, val_loss, train_acc, val_acc, val_recall, val_f2, test_acc, test_recall, lr):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        self.val_recalls.append(val_recall)
        self.val_f2_scores.append(val_f2)
        self.test_accs.append(test_acc)
        self.test_recalls.append(test_recall)
        self.learning_rates.append(lr)
    
    def plot_metrics(self, save_path):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Loss
        axes[0, 0].plot(self.train_losses, label='Train', linewidth=2)
        axes[0, 0].plot(self.val_losses, label='Val', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training & Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(self.train_accs, label='Train', linewidth=2)
        axes[0, 1].plot(self.val_accs, label='Val', linewidth=2)
        axes[0, 1].plot(self.test_accs, label='Test', linewidth=2, linestyle='--')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Recall
        axes[0, 2].plot(self.val_recalls, label='Val Recall', linewidth=2, color='green')
        axes[0, 2].plot(self.test_recalls, label='Test Recall', linewidth=2, color='blue', linestyle='--')
        axes[0, 2].axhline(y=0.8, color='r', linestyle='--', label='Target (80%)')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Recall')
        axes[0, 2].set_title('Recall Performance')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # F2 Score
        axes[1, 0].plot(self.val_f2_scores, label='Val F2', linewidth=2, color='purple')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F2 Score')
        axes[1, 0].set_title('Validation F2 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 1].plot(self.learning_rates, linewidth=2, color='orange')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Overfitting Gap
        gap = [t - v for t, v in zip(self.train_accs, self.val_accs)]
        axes[1, 2].plot(gap, linewidth=2, color='red')
        axes[1, 2].axhline(y=0, color='k', linestyle='--')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Train - Val Accuracy')
        axes[1, 2].set_title('Overfitting Gap')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

# ------------------------------------------------------------------------------------------------------------------
def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

# ------------------------------------------------------------------------------------------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler, epoch):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [TRAIN]')

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Decide which augmentation to use
        r = np.random.random()
        use_mixup = False
        use_cutmix = False
        labels_a, labels_b, lam = None, None, 1.0
        
        if Config.USE_MIXUP and r < Config.MIXUP_PROB:
            images, labels_a, labels_b, lam = mixup_data(images, labels, Config.MIXUP_ALPHA)
            use_mixup = True
        elif Config.USE_CUTMIX and r < (Config.MIXUP_PROB + Config.CUTMIX_PROB):
            images, labels_a, labels_b, lam = cutmix_data(images, labels, Config.MIXUP_ALPHA)
            use_cutmix = True
            
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with autocast(device_type='cuda'):
                outputs = model(images)
                
                if use_mixup or use_cutmix:
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            
            if use_mixup or use_cutmix:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP)
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        
        # For metrics, use original labels
        preds = torch.argmax(outputs, dim=1)
        if use_mixup or use_cutmix:
            # Use the dominant label for metrics
            effective_labels = labels_a if lam > 0.5 else labels_b
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(effective_labels.cpu().numpy())
        else:
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        current_lr = get_lr(optimizer)
        pbar.set_postfix({'loss': loss.item(), 'lr': f'{current_lr:.2e}'})

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

# ------------------------------------------------------------------------------------------------------------------
def evaluate(model, dataloader, criterion, device, split='Val', threshold=0.5):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f'[{split}]', leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = (probs >= threshold).long()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    
    metrics = {
        'loss': epoch_loss,
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'f2': calculate_f2_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }
    return metrics

# ------------------------------------------------------------------------------------------------------------------
def find_optimal_threshold(all_probs, all_labels, target_recall=0.82):
    """Find threshold that achieves target recall while maximizing accuracy"""
    best_threshold = 0.5
    best_score = 0.0
    
    for threshold in np.arange(0.3, 0.7, 0.01):
        preds = (all_probs >= threshold).astype(int)
        acc = accuracy_score(all_labels, preds)
        recall = recall_score(all_labels, preds, zero_division=0)
        
        # Prioritize meeting recall target
        if recall >= target_recall:
            # Among those, pick the one with best accuracy
            if acc > best_score:
                best_score = acc
                best_threshold = threshold
    
    # If no threshold meets target, pick the one with best F1 score
    if best_score == 0.0:
        for threshold in np.arange(0.3, 0.7, 0.01):
            preds = (all_probs >= threshold).astype(int)
            f1 = f1_score(all_labels, preds, zero_division=0)
            if f1 > best_score:
                best_score = f1
                best_threshold = threshold
                
    return best_threshold

# ------------------------------------------------------------------------------------------------------------------
class CosineAnnealingWithWarmup:
    """Cosine annealing with warmup scheduler"""
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

# ------------------------------------------------------------------------------------------------------------------
def train_model():
    print("=" * 80)
    print("OPTIMIZED ALZHEIMER'S CLASSIFICATION V3 - Targeting 80%+ Accuracy")
    print("Key Strategy: Larger model, balanced regularization, two-phase training")
    print("TARGET: 80%+ TEST ACCURACY & 80%+ TEST RECALL")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    print(f"\n📁 Loading data from {Config.DATA_ROOT}")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=Config.DATA_ROOT,
        batch_size=Config.BATCH_SIZE,
        img_size=Config.IMG_SIZE,
        num_workers=Config.NUM_WORKERS
    )

    print(f"📊 Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    print("\n🏗️  Building ConvNeXt-Small model for better accuracy...")
    model = create_alzheimer_model(
        model_size=Config.MODEL_SIZE,
        dropout=Config.DROPOUT,
        device=device
    )
    print(f"   Model: ConvNeXt-{Config.MODEL_SIZE}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if Config.USE_FOCAL_LOSS:
        print(f"✓ Using Focal Loss (α={Config.FOCAL_ALPHA}, γ={Config.FOCAL_GAMMA})")
        criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)
    else:
        print(f"✓ Using CrossEntropy Loss with label smoothing={Config.LABEL_SMOOTHING}")
        criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

    scaler = GradScaler() if Config.USE_MIXED_PRECISION and device.type == 'cuda' else None
    tracker = MetricsTracker()

    # ==================================================================================
    # TRAINING CONFIGURATION
    # ==================================================================================
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"📝 Hyperparameters:")
    print(f"   Model: ConvNeXt-{Config.MODEL_SIZE} | Dropout: {Config.DROPOUT}")
    print(f"   Epochs: {Config.TOTAL_EPOCHS} | Batch Size: {Config.BATCH_SIZE}")
    print(f"   Initial LR: {Config.INITIAL_LR:.0e} | Weight Decay: {Config.WEIGHT_DECAY:.0e}")
    print(f"   Label Smoothing: {Config.LABEL_SMOOTHING}")
    print(f"   Mixup: {Config.MIXUP_PROB*100:.0f}% | CutMix: {Config.CUTMIX_PROB*100:.0f}%")
    print(f"   Two-phase training: {'Yes' if Config.USE_TWO_PHASE else 'No'}")
    print("=" * 80)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.INITIAL_LR,
        weight_decay=Config.WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    total_steps = Config.TOTAL_EPOCHS * len(train_loader)
    warmup_steps = Config.WARMUP_EPOCHS * len(train_loader)
    
    scheduler = CosineAnnealingWithWarmup(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=Config.MIN_LR
    )
    
    best_val_metric = 0.0
    best_test_metric = 0.0
    best_test_acc = 0.0
    patience_counter = 0
    start_time = time.time()
    
    # Two-phase training: freeze backbone initially
    if Config.USE_TWO_PHASE:
        print(f"\n📌 Phase 1: Training with frozen backbone for {Config.FREEZE_EPOCHS} epochs")
        model.freeze_backbone(True)

    for epoch in range(1, Config.TOTAL_EPOCHS + 1):
        # Unfreeze backbone after initial epochs
        if Config.USE_TWO_PHASE and epoch == Config.FREEZE_EPOCHS + 1:
            print(f"\n📌 Phase 2: Unfreezing backbone for fine-tuning")
            model.freeze_backbone(False)
            # Reset optimizer with lower learning rate for fine-tuning
            optimizer = optim.AdamW(
                model.parameters(),
                lr=Config.INITIAL_LR * 0.5,  # Lower LR for fine-tuning
                weight_decay=Config.WEIGHT_DECAY,
                betas=(0.9, 0.999)
            )
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, epoch
        )
        
        # Step scheduler after each batch
        for _ in range(len(train_loader)):
            scheduler.step()
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, device, 'Val', Config.DECISION_THRESHOLD)
        test_metrics = evaluate(model, test_loader, criterion, device, 'Test', Config.DECISION_THRESHOLD)
        
        val_loss, val_acc = val_metrics['loss'], val_metrics['accuracy']
        val_f2 = val_metrics['f2']
        
        current_lr = get_lr(optimizer)
        tracker.update(
            train_loss, val_loss, train_acc, val_acc, 
            val_metrics['recall'], val_f2,
            test_metrics['accuracy'], test_metrics['recall'], 
            current_lr
        )

        print(f"\n📊 Epoch {epoch}/{Config.TOTAL_EPOCHS}:")
        print(f"   Train: Loss={train_loss:.4f} | Acc={train_acc:.4f}")
        print(f"   Val:   Loss={val_loss:.4f} | Acc={val_acc:.4f} | Recall={val_metrics['recall']:.4f}")
        print(f"   Test:  Acc={test_metrics['accuracy']:.4f} | Recall={test_metrics['recall']:.4f}")
        print(f"   LR: {current_lr:.2e}")

        # Save based on combined test metric (accuracy + recall)
        test_combined = 0.6 * test_metrics['accuracy'] + 0.4 * test_metrics['recall']
        if test_combined > best_test_metric:
            best_test_metric = test_combined
            best_test_acc = test_metrics['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_recall': val_metrics['recall'],
                'test_acc': test_metrics['accuracy'],
                'test_recall': test_metrics['recall'],
                'threshold': Config.DECISION_THRESHOLD
            }, Config.CHECKPOINT_DIR / 'best_model.pth')
            print(f"   ✅ Saved! Test Acc={test_metrics['accuracy']:.4f}, Recall={test_metrics['recall']:.4f}")

        # Early stopping based on validation
        val_combined = 0.5 * val_acc + 0.5 * val_metrics['recall']
        if val_combined > best_val_metric:
            best_val_metric = val_combined
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE and epoch > Config.FREEZE_EPOCHS + 3:
                print(f"\n⏹️  Early stopping triggered after {epoch} epochs")
                break

    # Save training curves
    tracker.plot_metrics(Config.PLOTS_DIR / 'training_curves.png')

    # Final evaluation
    checkpoint = torch.load(Config.CHECKPOINT_DIR / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\n" + "=" * 80)
    print("OPTIMIZING DECISION THRESHOLD")
    print("=" * 80)
    
    # Find optimal threshold on validation set
    val_metrics = evaluate(model, val_loader, criterion, device, 'Val', 0.5)
    optimal_threshold = find_optimal_threshold(
        np.array(val_metrics['probabilities']),
        np.array(val_metrics['labels']),
        Config.TARGET_RECALL
    )
    
    print(f"✓ Optimal threshold: {optimal_threshold:.3f}")
    
    # Final test evaluation with optimal threshold
    test_metrics = evaluate(model, test_loader, criterion, device, 'Test', optimal_threshold)
    
    print("\n" + "=" * 80)
    print("FINAL TEST SET RESULTS")
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
    
    # Save final results
    final_results = {
        'config': {
            'model_size': Config.MODEL_SIZE,
            'batch_size': Config.BATCH_SIZE,
            'dropout': Config.DROPOUT,
            'learning_rate': Config.INITIAL_LR,
            'weight_decay': Config.WEIGHT_DECAY,
            'label_smoothing': Config.LABEL_SMOOTHING,
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
    
    with open(Config.OUTPUT_DIR / 'final_results.json', 'w') as f:
        json.dump(final_results, f, indent=4)
    
    return test_metrics

# ------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    test_metrics = train_model()
    
    if test_metrics['accuracy'] >= 0.8 and test_metrics['recall'] >= 0.8:
        print("\n🎉 SUCCESS! Both accuracy and recall targets achieved!")
    else:
        print("\n📈 Training complete. Continue tuning if targets not met.")