"""
Training script for GFNet on ADNI Alzheimer's classification
Optimized for achieving 80%+ accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import time
import json
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, roc_auc_score)

from dataset import create_dataloaders
from modules import create_gfnet_model


# ==================================================================================
# CONFIGURATION
# ==================================================================================
class Config:
    # Data
    DATA_ROOT = './data/ADNI/AD_NC'
    
    # Model
    IMG_SIZE = 210
    PATCH_SIZE = 14
    EMBED_DIM = 384
    DEPTH = 12
    
    # Training
    BATCH_SIZE = 8
    ACCUMULATION_STEPS = 8  # Effective batch size = 64
    EPOCHS = 40
    INITIAL_LR = 1e-3  # Reduced from 1e-3
    WEIGHT_DECAY = 1e-3  # Increased from 5e-4
    
    
    # Loss
    LABEL_SMOOTHING = 0.1
    
    # Early stopping
    PATIENCE = 20  # Reduced from 2000
    MIN_DELTA = 0.001
    
    # Learning rate scheduler (CosineAnnealingWarmRestarts)
    LR_T0 = 10  # Initial restart period
    LR_T_MULT = 2  # Multiply period by 2 after each restart
    LR_ETA_MIN = 1e-7  # Minimum learning rate

    # Other
    NUM_WORKERS = 4
    VAL_SPLIT = 0.1
    USE_CLAHE = False  # Disabled to avoid distribution mismatch
    
    # Output
    OUTPUT_DIR = Path('./gfnet_results')
    CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'
    PLOTS_DIR = OUTPUT_DIR / 'plots'
    
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)


# ==================================================================================
# TRAINING FUNCTIONS
# ==================================================================================
def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler, epoch, accumulation_steps=1):
    """Train for one epoch with gradient accumulation"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # Track per-class metrics
    class_correct = [0, 0]
    class_total = [0, 0]

    optimizer.zero_grad()  # Zero gradients at start

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        # Mixed precision training
        with autocast(device_type=device.type):
            outputs = model(images)
            loss = criterion(outputs, labels) / accumulation_steps  # Scale loss

        # Backward pass (accumulate gradients)
        scaler.scale(loss).backward()

        # Gradient accumulation: only step optimizer every N batches
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Reduced from 1.0
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Metrics (accumulate over all batches, not just accumulation steps)
        total_loss += loss.item() * accumulation_steps  # Rescale for proper averaging
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Per-class accuracy
        for i in range(2):
            mask = (labels == i)
            class_total[i] += mask.sum().item()
            class_correct[i] += ((predicted == labels) & mask).sum().item()

        # Progress bar
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{loss.item()*accumulation_steps:.4f}',
            'acc': f'{100.*correct/total:.2f}%',
            'lr': f'{current_lr:.6f}'
        })

    epoch_loss = total_loss / len(dataloader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    for images, labels in tqdm(dataloader, desc='Evaluating', leave=False):
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())
    
    # Calculate metrics
    metrics = {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0,
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels)
    }
    
    return metrics


def plot_training_history(history, save_path):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[0, 1].axhline(y=0.8, color='g', linestyle='--', label='Target (80%)', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Recall
    axes[1, 0].plot(epochs, history['val_recall'], 'purple', linewidth=2, marker='o')
    axes[1, 0].axhline(y=0.7, color='g', linestyle='--', label='Target (70%)', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Recall', fontsize=12)
    axes[1, 0].set_title('Validation Recall (AD Class)', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # F1 Score
    axes[1, 1].plot(epochs, history['val_f1'], 'orange', linewidth=2, marker='s')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('F1 Score', fontsize=12)
    axes[1, 1].set_title('Validation F1 Score', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6) * 100
    
    plt.figure(figsize=(10, 8))
    
    labels = np.array([[f"{cm[i,j]}\n({cm_percent[i,j]:.1f}%)"
                       for j in range(2)] for i in range(2)])
    
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues',
                xticklabels=['Predicted: NC', 'Predicted: AD'],
                yticklabels=['True: NC', 'True: AD'],
                linewidths=2, linecolor='black',
                annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    
    # Add metrics
    tn, fp, fn, tp = cm.ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    textstr = f'AD Recall: {recall:.3f}\nAD Precision: {precision:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    plt.text(1.3, 0.5, textstr, transform=plt.gca().transAxes,
            fontsize=12, verticalalignment='center', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ==================================================================================
# MAIN TRAINING LOOP
# ==================================================================================
def train_model():
    """Main training function"""
    print("\n" + "="*80)
    print("GFNET TRAINING FOR ALZHEIMER'S DISEASE CLASSIFICATION")
    print("Target: 80%+ Accuracy")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️ Device: {device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    train_loader, val_loader, test_loader = create_dataloaders(
        Config.DATA_ROOT,
        batch_size=Config.BATCH_SIZE,
        img_size=Config.IMG_SIZE,
        num_workers=Config.NUM_WORKERS,
        val_split=Config.VAL_SPLIT,
        use_clahe=Config.USE_CLAHE
    )
    
    # Create model
    model = create_gfnet_model(
        img_size=Config.IMG_SIZE,
        patch_size=Config.PATCH_SIZE,
        in_chans=1,
        num_classes=2,
        embed_dim=Config.EMBED_DIM,
        depth=Config.DEPTH,
        drop_rate=0.2,  # Increased from 0.1 for regularization
        drop_path_rate=0.2  # Increased from 0.1 for regularization
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(),
                           lr=Config.INITIAL_LR,
                           weight_decay=Config.WEIGHT_DECAY)

    # Learning rate scheduler - Warmup + CosineAnnealingWarmRestarts
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=5  # 5 epoch warmup
    )

    main_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=Config.LR_T0,
        T_mult=Config.LR_T_MULT,
        eta_min=Config.LR_ETA_MIN
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[5]  # Switch from warmup to main scheduler after 5 epochs
    )

    
    # Mixed precision
    scaler = GradScaler()
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 
        'val_recall': [], 'val_f1': []
    }
    
    best_val_acc = 0
    patience_counter = 0
    start_time = time.time()
    
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, epoch, Config.ACCUMULATION_STEPS
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        
        # Print progress
        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
        print(f"\n📊 Epoch {epoch}/{Config.EPOCHS} (LR: {current_lr:.6f}):")
        print(f"   Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        print(f"   Val:   Loss={val_metrics['loss']:.4f}, Acc={val_metrics['accuracy']:.4f}")
        print(f"   📌 Recall={val_metrics['recall']:.4f}, F1={val_metrics['f1']:.4f}")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            print(f"   ✅ New best validation accuracy! Saving model...")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_metrics['accuracy'],
                'val_recall': val_metrics['recall']
            }, Config.CHECKPOINT_DIR / 'best_model.pth')

            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= Config.PATIENCE:
            print(f"\n⚠️ Early stopping at epoch {epoch}")
            break
    
    training_time = time.time() - start_time
    print(f"\n⏱️ Training completed in {training_time/60:.1f} minutes")
    
    # Plot training history
    plot_training_history(history, Config.PLOTS_DIR / 'training_history.png')
    print(f"📊 Training plots saved")
    
    # ==================================================================================
    # TEST EVALUATION
    # ==================================================================================
    print("\n" + "="*80)
    print("FINAL TEST EVALUATION")
    print("="*80)
    
    # Load best model
    checkpoint = torch.load(Config.CHECKPOINT_DIR / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded best model from epoch {checkpoint['epoch']}")
    
    # Test
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    print("\n" + "="*80)
    print("🎯 TEST RESULTS:")
    print("="*80)
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print(f"  AUC:       {test_metrics['auc']:.4f}")
    print("="*80)
    
    # Save results
    results = {
        'test_accuracy': float(test_metrics['accuracy']),
        'test_precision': float(test_metrics['precision']),
        'test_recall': float(test_metrics['recall']),
        'test_f1': float(test_metrics['f1']),
        'test_auc': float(test_metrics['auc']),
        'training_time_minutes': training_time / 60,
        'epochs_trained': len(history['train_loss'])
    }
    
    with open(Config.OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        test_metrics['labels'],
        test_metrics['predictions'],
        Config.PLOTS_DIR / 'confusion_matrix.png'
    )
    
    print(f"\n✅ All results saved to {Config.OUTPUT_DIR}")
    
    if test_metrics['accuracy'] >= 0.80:
        print("\n" + "="*80)
        print("🎉 SUCCESS! Achieved 80%+ accuracy! 🎉")
        print("="*80)
    
    return test_metrics


if __name__ == '__main__':
    train_model()