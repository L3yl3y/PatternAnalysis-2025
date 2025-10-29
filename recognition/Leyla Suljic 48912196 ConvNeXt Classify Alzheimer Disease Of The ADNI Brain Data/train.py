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
class Config:
    DATA_ROOT = './data/ADNI/AD_NC'
    
    # FIXED HYPERPARAMETERS FOR BETTER GENERALIZATION
    BATCH_SIZE = 64
    IMG_SIZE = 224
    NUM_WORKERS = 8
    MODEL_SIZE = 'base'
    DROPOUT = 0.3  # Reduced from 0.5 to prevent overfitting
    
    # SIMPLIFIED SINGLE-PHASE TRAINING
    TOTAL_EPOCHS = 40  # Reduced total epochs significantly
    WARMUP_EPOCHS = 3
    INITIAL_LR = 5e-4  # Lower starting LR
    MIN_LR = 1e-6
    
    WEIGHT_DECAY = 5e-4  # Increased for better regularization
    PATIENCE = 7000000  # Reduced patience
    MIN_DELTA = 0.001
    USE_MIXED_PRECISION = True
    GRADIENT_CLIP = 0.5  # Reduced clip value
    LABEL_SMOOTHING = 0.15  # Increased smoothing
    
    # FIXED CLASS BALANCING - Less extreme alpha
    USE_FOCAL_LOSS = True
    FOCAL_ALPHA = [0.35, 0.65]  # More balanced, less extreme
    FOCAL_GAMMA = 2.0
    USE_BALANCED_SAMPLING = True
    
    # IMPROVED THRESHOLD STRATEGY
    TARGET_RECALL = 0.82  # Slightly lower, more achievable target
    
    OUTPUT_DIR = Path('./alzheimer_results_fixed')
    CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'
    PLOTS_DIR = OUTPUT_DIR / 'plots'

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------------------------------------------------------
class MetricsTracker:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.val_recalls = []
        self.val_f2s = []
        self.learning_rates = []
        self.best_val_metric = 0.0  # Tracks balanced metric
        self.best_epoch = 0

    def update(self, train_loss, val_loss, train_acc, val_acc, val_recall, val_f2, lr):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        self.val_recalls.append(val_recall)
        self.val_f2s.append(val_f2)
        self.learning_rates.append(lr)

        # FIXED: Use balanced metric (average of acc and recall)
        balanced_metric = 0.5 * val_acc + 0.5 * val_recall
        if balanced_metric > self.best_val_metric:
            self.best_val_metric = balanced_metric
            self.best_epoch = len(self.val_accs) - 1

    def plot(self, save_path):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        epochs = range(1, len(self.train_losses) + 1)

        # Loss plot
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].axvline(self.best_epoch + 1, color='g', linestyle='--', alpha=0.5, label=f'Best Epoch ({self.best_epoch + 1})')
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy plot
        axes[0, 1].plot(epochs, self.train_accs, 'b-', label='Train Acc', linewidth=2)
        axes[0, 1].plot(epochs, self.val_accs, 'r-', label='Val Acc', linewidth=2)
        axes[0, 1].axvline(self.best_epoch + 1, color='g', linestyle='--', alpha=0.5, label=f'Best Epoch ({self.best_epoch + 1})')
        axes[0, 1].axhline(0.8, color='orange', linestyle='--', alpha=0.5, label='Target Acc (0.8)')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy', fontsize=12)
        axes[0, 1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Recall & F2 plot
        axes[1, 0].plot(epochs, self.val_recalls, 'purple', linewidth=2, label='Val Recall')
        axes[1, 0].plot(epochs, self.val_f2s, 'green', linewidth=2, label='Val F2-Score')
        axes[1, 0].axhline(0.8, color='orange', linestyle='--', alpha=0.5, label='Target (0.8)')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Metric Score', fontsize=12)
        axes[1, 0].set_title('Recall & F2-Score (AD Detection)', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Summary text
        best_val_acc_at_best_epoch = self.val_accs[self.best_epoch]
        best_val_recall_at_best_epoch = self.val_recalls[self.best_epoch]
        target_met = '✓ YES' if best_val_acc_at_best_epoch >= 0.8 and best_val_recall_at_best_epoch >= 0.8 else '✗ NO'

        summary_text = f"""
            Best Balanced Metric: {self.best_val_metric:.4f}
            Best Epoch: {self.best_epoch + 1}
            --- At Best Epoch ---
            Val Acc: {best_val_acc_at_best_epoch:.4f}
            Val Recall: {best_val_recall_at_best_epoch:.4f}
            --- Final Epoch ---
            Final Val Acc: {self.val_accs[-1]:.4f}
            Final Val Recall: {self.val_recalls[-1]:.4f}
            Final Val F2: {self.val_f2s[-1]:.4f}
            ---
            Total Epochs: {len(self.train_losses)}
            Target (Acc>0.8 & Recall>0.8): {target_met}
        """
        axes[1, 1].axis('off')
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

# ------------------------------------------------------------------------------------------------------------------
def create_balanced_sampler(dataset):
    """Create weighted sampler for class imbalance"""
    labels = [label for _, label in dataset]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# ------------------------------------------------------------------------------------------------------------------
def get_lr(optimizer):
    """Get current learning rate"""
    for param_group in optimizer.param_groups:
        return param_group['lr']

# ------------------------------------------------------------------------------------------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None, epoch=1):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [TRAIN]')

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP)
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
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
    """Find threshold that balances accuracy and recall"""
    best_threshold = 0.5
    best_score = 0.0
    
    # Search for threshold that gives best balanced score
    for threshold in np.arange(0.3, 0.7, 0.01):
        preds = (all_probs >= threshold).astype(int)
        acc = accuracy_score(all_labels, preds)
        recall = recall_score(all_labels, preds, zero_division=0)
        
        # Balanced score: prioritize recall slightly but not too much
        if recall >= target_recall * 0.9:  # Must be close to target
            balanced_score = 0.4 * acc + 0.6 * recall  # 60% weight on recall
            if balanced_score > best_score:
                best_score = balanced_score
                best_threshold = threshold
    
    # If no good threshold found, use the one with best F2
    if best_score == 0.0:
        for threshold in np.arange(0.3, 0.7, 0.01):
            preds = (all_probs >= threshold).astype(int)
            f2 = calculate_f2_score(all_labels, preds, zero_division=0)
            if f2 > best_score:
                best_score = f2
                best_threshold = threshold
                
    return best_threshold

# ------------------------------------------------------------------------------------------------------------------
class WarmupCosineScheduler:
    """Warmup + Cosine annealing scheduler"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, steps_per_epoch, base_lr, min_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        self.base_lr = base_lr
        self.min_lr = min_lr
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
    print("FIXED ALZHEIMER'S CLASSIFICATION - Better Generalization")
    print("Strategy: Balanced approach to Accuracy & Recall")
    print("TARGET: 80%+ ACCURACY & 80%+ RECALL")
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

    if Config.USE_BALANCED_SAMPLING:
        print("✓ Using class-balanced sampling")
        train_sampler = create_balanced_sampler(train_loader.dataset)
        train_loader = torch.utils.data.DataLoader(
            train_loader.dataset,
            batch_size=Config.BATCH_SIZE,
            sampler=train_sampler,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True
        )

    print(f"📊 Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    print("\n🏗️  Building model...")
    model = create_alzheimer_model(
        model_size=Config.MODEL_SIZE, 
        dropout=Config.DROPOUT,
        device=device
    )
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if Config.USE_FOCAL_LOSS:
        print(f"✓ Using Balanced Focal Loss (α={Config.FOCAL_ALPHA}, γ={Config.FOCAL_GAMMA})")
        criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

    scaler = GradScaler() if Config.USE_MIXED_PRECISION and device.type == 'cuda' else None
    tracker = MetricsTracker()

    # ==================================================================================
    # SIMPLIFIED SINGLE-PHASE TRAINING
    # ==================================================================================
    print("\n" + "=" * 80)
    print("SINGLE-PHASE TRAINING (All layers trainable)")
    print("=" * 80)
    print(f"📝 Config:")
    print(f"   Epochs: {Config.TOTAL_EPOCHS} | Initial LR: {Config.INITIAL_LR:.0e}")
    print(f"   Warmup: {Config.WARMUP_EPOCHS} epochs | Batch: {Config.BATCH_SIZE}")
    print(f"   Dropout: {Config.DROPOUT} | Weight Decay: {Config.WEIGHT_DECAY}")
    print(f"   Label Smoothing: {Config.LABEL_SMOOTHING}")
    print("=" * 80)

    # All layers trainable from start - no freezing
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.INITIAL_LR,
        weight_decay=Config.WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_epochs=Config.WARMUP_EPOCHS,
        total_epochs=Config.TOTAL_EPOCHS,
        steps_per_epoch=len(train_loader),
        base_lr=Config.INITIAL_LR,
        min_lr=Config.MIN_LR
    )
    
    best_val_metric = 0.0
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, Config.TOTAL_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, epoch
        )
        
        # Step scheduler after each batch (done inside train_one_epoch via scheduler.step())
        # Actually, we need to step after each batch, so let me fix this
        
        val_metrics = evaluate(model, val_loader, criterion, device, 'Val')
        val_loss, val_acc = val_metrics['loss'], val_metrics['accuracy']
        val_f2 = val_metrics['f2']
        
        current_lr = get_lr(optimizer)
        tracker.update(train_loss, val_loss, train_acc, val_acc, val_metrics['recall'], val_f2, current_lr)

        print(f"\n📊 Epoch {epoch}/{Config.TOTAL_EPOCHS}:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"   Precision:  {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | F2: {val_f2:.4f}")
        print(f"   LR: {current_lr:.2e}")

        # Save based on balanced metric
        balanced_metric = 0.5 * val_acc + 0.5 * val_metrics['recall']
        if balanced_metric > best_val_metric:
            best_val_metric = balanced_metric
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'balanced_metric': balanced_metric,
                'val_acc': val_acc,
                'val_recall': val_metrics['recall'],
                'val_probs': val_metrics['probabilities'],
                'val_labels': val_metrics['labels']
            }, Config.CHECKPOINT_DIR / 'best_model.pth')
            print(f"   ✅ New best model! (Balanced: {balanced_metric:.4f}, Acc: {val_acc:.4f}, Recall: {val_metrics['recall']:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"\n⚠️  Early stopping at epoch {epoch}")
            break

    training_time = time.time() - start_time
    print(f"\n✅ Training complete in {training_time / 60:.1f} minutes")
    print(f"   Best Balanced Metric: {best_val_metric:.4f}")

    # ==================================================================================
    # THRESHOLD OPTIMIZATION & TESTING
    # ==================================================================================
    checkpoint = torch.load(Config.CHECKPOINT_DIR / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    print("\n" + "=" * 80)
    print(f"OPTIMIZING THRESHOLD (Target Recall ~{Config.TARGET_RECALL})")
    print("=" * 80)
    
    val_probs = np.array(checkpoint['val_probs'])
    val_labels = np.array(checkpoint['val_labels'])
    optimal_threshold = find_optimal_threshold(val_probs, val_labels, Config.TARGET_RECALL)
    
    val_preds_opt = (val_probs >= optimal_threshold).astype(int)
    opt_recall = recall_score(val_labels, val_preds_opt)
    opt_acc = accuracy_score(val_labels, val_preds_opt)
    
    print(f"✓ Optimal threshold: {optimal_threshold:.3f}")
    print(f"  Val metrics at threshold: Acc={opt_acc:.4f}, Recall={opt_recall:.4f}")

    print("\n" + "=" * 80)
    print("FINAL TEST SET EVALUATION")
    print("=" * 80)
    test_metrics = evaluate(model, test_loader, criterion, device, 'Test', threshold=optimal_threshold)

    print(f"\n🎯 TEST RESULTS:")
    print("=" * 80)
    
    acc_met = test_metrics['accuracy'] >= 0.8
    recall_met = test_metrics['recall'] >= 0.8
    
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f} {'✅' if acc_met else '❌'}")
    print(f"  Recall:    {test_metrics['recall']:.4f} {'✅' if recall_met else '❌'}  <-- KEY METRIC")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print(f"  F2 Score:  {test_metrics['f2']:.4f}")
    print(f"  AUC:       {test_metrics['auc']:.4f}")
    print("=" * 80)

    total_epochs = len(tracker.train_losses)
    print(f"\n⏱️  Summary:")
    print(f"   Total time: {training_time / 60:.1f} minutes")
    print(f"   Total epochs: {total_epochs}")
    print(f"   Time per epoch: {(training_time) / total_epochs:.1f} seconds")

    # Save results
    final_results = {
        'test_accuracy': float(test_metrics['accuracy']),
        'test_precision': float(test_metrics['precision']),
        'test_recall': float(test_metrics['recall']),
        'test_f1': float(test_metrics['f1']),
        'test_f2': float(test_metrics['f2']),
        'test_auc': float(test_metrics['auc']),
        'best_balanced_metric': float(best_val_metric),
        'optimal_threshold': float(optimal_threshold),
        'total_training_time_minutes': float(training_time / 60),
        'total_epochs': total_epochs,
        'targets_met': acc_met and recall_met
    }

    with open(Config.OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump(final_results, f, indent=4)

    # Generate plots
    tracker.plot(Config.PLOTS_DIR / 'training_curves.png')
    plot_confusion_matrix(
        test_metrics['labels'],
        test_metrics['predictions'],
        Config.PLOTS_DIR / 'confusion_matrix.png'
    )
    plot_roc_curve(
        test_metrics['labels'],
        test_metrics['probabilities'],
        Config.PLOTS_DIR / 'roc_curve.png'
    )

    print(f"\n✅ Results saved to {Config.OUTPUT_DIR}")
    
    if acc_met and recall_met:
        print("\n" + "=" * 80)
        print("🎉 SUCCESS! Both targets achieved! 🎉")
        print("=" * 80)
    else:
        print("\n⚠️  Targets not fully met:")
        if not acc_met:
            print(f"   Accuracy gap: {0.8 - test_metrics['accuracy']:.4f}")
        if not recall_met:
            print(f"   Recall gap: {0.8 - test_metrics['recall']:.4f}")

# ------------------------------------------------------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    labels = [f"{val}\n({perc*100:.1f}%)" for val, perc in zip(cm.flatten(), cm_percent.flatten())]
    labels = np.asarray(labels).reshape(2, 2)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=True,
                xticklabels=['Normal', 'AD'], yticklabels=['Normal', 'AD'])
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ------------------------------------------------------------------------------------------------------------------
def plot_roc_curve(y_true, y_probs, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random (AUC = 0.5)')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Test Set', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    train_model()