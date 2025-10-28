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
# Preface: this code isn't all too exciting a lot of it does involve a lot of printing because I like being able to
# see my metrics in a human readable format :3
# ------------------------------------------------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha = 0.75, gamma = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction = 'none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# ------------------------------------------------------------------------------------------------------------------
class Config:
    DATA_ROOT = './data/ADNI/AD_NC'
    BATCH_SIZE = 16
    IMG_SIZE = 224
    NUM_WORKERS = 8
    MODEL_SIZE = 'base'
    DROPOUT = 0.4

    PHASE1_EPOCHS = 15
    PHASE1_LR = 1e-3
    
    PHASE2_EPOCHS = 50
    PHASE2_LR = 5e-6
    WEIGHT_DECAY = 1e-4
    PATIENCE = 15
    MIN_DELTA = 0.001
    USE_MIXED_PRECISION = True
    GRADIENT_CLIP = 1.0
    LABEL_SMOOTHING = 0.05
    CLASS_WEIGHTS = [0.4, 2.5]
    
    USE_FOCAL_LOSS = True
    FOCAL_ALPHA = 0.75
    FOCAL_GAMMA = 2.0
    USE_BALANCED_SAMPLING = True
    OPTIMIZE_THRESHOLD = True
    TARGET_RECALL = 0.85

    OUTPUT_DIR = Path('./alzheimer_results')
    CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'
    PLOTS_DIR = OUTPUT_DIR / 'plots'

    OUTPUT_DIR.mkdir(exist_ok = True, parents = True)
    CHECKPOINT_DIR.mkdir(exist_ok = True)
    PLOTS_DIR.mkdir(exist_ok = True)

# ------------------------------------------------------------------------------------------------------------------
class MetricsTracker:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.val_recalls = []
        self.learning_rates = []
        self.best_val_acc = 0.0
        self.best_epoch = 0

    def update(self, train_loss, val_loss, train_acc, val_acc, val_recall, lr):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        self.val_recalls.append(val_recall)
        self.learning_rates.append(lr)

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = len(self.val_accs) - 1

    def plot(self, save_path):
        fig, axes = plt.subplots(2, 2, figsize = (15, 10))
        epochs = range(1, len(self.train_losses) + 1)

        # ------------------------------------------------------------------------------------------------------------------
        # PLOT 1: Loss plot.
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label = 'Train Loss', linewidth = 2)
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label = 'Val Loss', linewidth = 2)
        axes[0, 0].axvline(self.best_epoch + 1, color = 'g', linestyle = '--', alpha = 0.5, label = f'Best Epoch ({self.best_epoch + 1})')
        axes[0, 0].set_xlabel('Epoch', fontsize = 12)
        axes[0, 0].set_ylabel('Loss', fontsize = 12)
        axes[0, 0].set_title('Training & Validation Loss', fontsize = 14, fontweight = 'bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha = 0.3)

        # ------------------------------------------------------------------------------------------------------------------
        # PLOT 2: Accuracy plot.
        axes[0, 1].plot(epochs, self.train_accs, 'b-', label = 'Train Acc', linewidth = 2)
        axes[0, 1].plot(epochs, self.val_accs, 'r-', label = 'Val Acc', linewidth = 2)
        axes[0, 1].axvline(self.best_epoch + 1, color = 'g', linestyle = '--', alpha = 0.5, label = f'Best Epoch ({self.best_epoch + 1})')
        axes[0, 1].axhline(0.8, color = 'orange', linestyle = '--', alpha = 0.5, label = 'Target (0.8)')
        axes[0, 1].axhline(0.85, color = 'purple', linestyle = '--', alpha = 0.5, label = 'Goal (0.85)')
        axes[0, 1].set_xlabel('Epoch', fontsize = 12)
        axes[0, 1].set_ylabel('Accuracy', fontsize = 12)
        axes[0, 1].set_title('Training & Validation Accuracy', fontsize = 14, fontweight = 'bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha = 0.3)

        # ------------------------------------------------------------------------------------------------------------------
        # PLOT 3: Recall (AD Detection).
        axes[1, 0].plot(epochs, self.val_recalls, 'purple', linewidth = 2, label = 'Val Recall')
        axes[1, 0].axhline(0.85, color = 'orange', linestyle = '--', alpha = 0.5, label = 'Target (0.85)')
        axes[1, 0].set_xlabel('Epoch', fontsize = 12)
        axes[1, 0].set_ylabel('Recall (AD Detection)', fontsize = 12)
        axes[1, 0].set_title('AD Detection Rate', fontsize = 14, fontweight = 'bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha = 0.3)

        # ------------------------------------------------------------------------------------------------------------------
        # PLOT 4: Summary text.
        summary_text = f"""
            Best Validation Accuracy: {self.best_val_acc:.4f}
            Best Epoch: {self.best_epoch + 1}
            Final Train Acc: {self.train_accs[-1]:.4f}
            Final Val Acc: {self.val_accs[-1]:.4f}
            Final Val Recall: {self.val_recalls[-1]:.4f}
            Final Train Loss: {self.train_losses[-1]:.4f}
            Final Val Loss: {self.val_losses[-1]:.4f}
            Total Epochs: {len(self.train_losses)}
            Target Achieved: {'✓ YES' if self.best_val_acc >= 0.8 else '✗ NO'}
        """
        axes[1, 1].axis('off')
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize = 11, verticalalignment = 'center', fontfamily = 'monospace', bbox = dict(boxstyle = 'round', facecolor = 'wheat', alpha = 0.3))
        plt.tight_layout()
        plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
        plt.close()

# ------------------------------------------------------------------------------------------------------------------
def create_balanced_sampler(dataset):
    labels = [label for _, label in dataset]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(weights = sample_weights, num_samples = len(sample_weights), replacement = True)

# ------------------------------------------------------------------------------------------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler = None, epoch = 1, phase = 'Phase1'):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    pbar = tqdm(dataloader, desc = f'{phase} Epoch {epoch} [TRAIN]')

    for images, labels in pbar:
        images = images.to(device, non_blocking = True)
        labels = labels.to(device, non_blocking = True)
        optimizer.zero_grad(set_to_none = True)

        if scaler is not None:
            with autocast(device_type = 'cuda'):
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
        preds = torch.argmax(outputs, dim = 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix({'loss': loss.item()})

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

# ------------------------------------------------------------------------------------------------------------------
def evaluate(model, dataloader, criterion, device, split = 'Val', threshold = 0.5):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc = f'[{split}]'):
            images = images.to(device, non_blocking = True)
            labels = labels.to(device, non_blocking = True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim = 1)[:, 1]
            preds = (probs >= threshold).long()
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = {
        'loss': epoch_loss,
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division = 0),
        'recall': recall_score(all_labels, all_preds, zero_division = 0),
        'f1': f1_score(all_labels, all_preds, zero_division = 0),
        'auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }
    return metrics

# ------------------------------------------------------------------------------------------------------------------
def find_optimal_threshold(model, dataloader, device, target_recall = 0.85):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc = '[THRESHOLD OPT]'):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim = 1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    best_threshold = 0.5
    best_acc = 0.0

    for threshold in np.arange(0.3, 0.7, 0.01):
        preds = (all_probs >= threshold).astype(int)
        recall = recall_score(all_labels, preds, zero_division = 0)
        if recall >= target_recall:
            acc = accuracy_score(all_labels, preds)
            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold

    print(f"\nOptimal threshold: {best_threshold:.3f}")
    print(f"  Achieves recall: {recall_score(all_labels, (all_probs >= best_threshold).astype(int)):.3f}")
    print(f"  With accuracy: {best_acc:.3f}")
    return best_threshold

# ------------------------------------------------------------------------------------------------------------------
def train_model():
    print("=" * 80)
    print("OPTIMIZED ALZHEIMER'S CLASSIFICATION")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nLoading data from {Config.DATA_ROOT}")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root = Config.DATA_ROOT,
        batch_size = Config.BATCH_SIZE,
        img_size = Config.IMG_SIZE,
        num_workers = Config.NUM_WORKERS
    )

    if Config.USE_BALANCED_SAMPLING:
        print("✓ Using class-balanced sampling")
        train_sampler = create_balanced_sampler(train_loader.dataset)
        train_loader = torch.utils.data.DataLoader(
            train_loader.dataset,
            batch_size = Config.BATCH_SIZE,
            sampler = train_sampler,
            num_workers = Config.NUM_WORKERS
        )

    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    print("\nBuilding model...")
    model = create_alzheimer_model(model_size = Config.MODEL_SIZE, dropout = Config.DROPOUT, device = device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if Config.USE_FOCAL_LOSS:
        print(f"✓ Using Focal Loss (alpha = {Config.FOCAL_ALPHA}, gamma = {Config.FOCAL_GAMMA})")
        criterion = FocalLoss(alpha = Config.FOCAL_ALPHA, gamma = Config.FOCAL_GAMMA)
    else:
        class_weights = torch.FloatTensor(Config.CLASS_WEIGHTS).to(device)
        criterion = nn.CrossEntropyLoss(weight = class_weights, label_smoothing = Config.LABEL_SMOOTHING)

    scaler = GradScaler() if Config.USE_MIXED_PRECISION and device.type == 'cuda' else None
    tracker = MetricsTracker()

    # ------------------------------------------------------------------------------------------------------------------
    # PHASE 1: FROZEN STUFF PHASE 1 BTW:
    # ------------------------------------------------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 1: CLASSIFIER HEAD TRAINING (FROZEN BACKBONE)")
    print("=" * 80)
    print(f"Epochs: {Config.PHASE1_EPOCHS} | LR: {Config.PHASE1_LR}")
    print("=" * 80)

    model.freeze_backbone(freeze = True)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr = Config.PHASE1_LR, weight_decay = Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = Config.PHASE1_EPOCHS, eta_min = 1e-6)
    best_val_acc = 0.0
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, Config.PHASE1_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, epoch, 'Phase1')
        val_metrics = evaluate(model, val_loader, criterion, device, 'Val')
        val_loss, val_acc = val_metrics['loss'], val_metrics['accuracy']
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        tracker.update(train_loss, val_loss, train_acc, val_acc, val_metrics['recall'], current_lr)

        print(f"\nEpoch {epoch}/{Config.PHASE1_EPOCHS}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"  Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f}")
        print(f"  LR: {current_lr:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc
            }, Config.CHECKPOINT_DIR / 'best_model_phase1.pth')
            print(f"  ✓ New best model saved! (Val Acc: {val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"\n⚠ Early stopping triggered after {epoch} epochs")
            break

    phase1_time = time.time() - start_time
    print(f"\n✓ Phase 1 complete in {phase1_time / 60:.1f} minutes")
    print(f"  Best Val Acc: {best_val_acc:.4f}")

    # ------------------------------------------------------------------------------------------------------------------
    # PHASE 2: FINE TUNE STUFF PHASE 2 UNFROZEN BTW:
    # Super similar to phase 1.
    # ------------------------------------------------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 2: FULL MODEL FINE-TUNING (UNFROZEN BACKBONE)")
    print("=" * 80)
    print(f"Epochs: {Config.PHASE2_EPOCHS} | LR: {Config.PHASE2_LR}")
    print("=" * 80)

    checkpoint = torch.load(Config.CHECKPOINT_DIR / 'best_model_phase1.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded best Phase 1 model (Val Acc: {checkpoint['val_acc']:.4f})")

    model.freeze_backbone(freeze = False)
    optimizer = optim.AdamW(model.parameters(), lr = Config.PHASE2_LR, weight_decay = Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = Config.PHASE2_EPOCHS, eta_min = 1e-7)
    best_val_acc = checkpoint['val_acc']
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, Config.PHASE2_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, epoch, 'Phase2')
        val_metrics = evaluate(model, val_loader, criterion, device, 'Val')
        val_loss, val_acc = val_metrics['loss'], val_metrics['accuracy']
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        tracker.update(train_loss, val_loss, train_acc, val_acc, val_metrics['recall'], current_lr)

        print(f"\nEpoch {epoch}/{Config.PHASE2_EPOCHS}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"  Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f}")
        print(f"  LR: {current_lr:.2e}")

        if val_acc > best_val_acc + Config.MIN_DELTA:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc
            }, Config.CHECKPOINT_DIR / 'best_model_final.pth')
            print(f"  ✓ New best model saved! (Val Acc: {val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"\n⚠ Early stopping triggered after {epoch} epochs")
            break

    phase2_time = time.time() - start_time
    print(f"\n✓ Phase 2 complete in {phase2_time / 60:.1f} minutes")
    print(f"  Best Val Acc: {best_val_acc:.4f}")

    # ------------------------------------------------------------------------------------------------------------------
    checkpoint = torch.load(Config.CHECKPOINT_DIR / 'best_model_final.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    optimal_threshold = 0.5
    if Config.OPTIMIZE_THRESHOLD:
        print("\n" + "=" * 80)
        print("OPTIMIZING DECISION THRESHOLD")
        print("=" * 80)
        optimal_threshold = find_optimal_threshold(model, val_loader, device, Config.TARGET_RECALL)

    # ------------------------------------------------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 80)
    print(f"✓ Using threshold: {optimal_threshold:.3f}")
    test_metrics = evaluate(model, test_loader, criterion, device, 'Test', threshold = optimal_threshold)

    print(f"\n🎯 TEST SET RESULTS:")
    print("=" * 80)
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f} {'✓ TARGET ACHIEVED!' if test_metrics['accuracy'] >= 0.8 else '✗ Below target'}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print(f"  AUC:       {test_metrics['auc']:.4f}")
    print("=" * 80)

    final_results = {
        'test_accuracy': float(test_metrics['accuracy']),
        'test_precision': float(test_metrics['precision']),
        'test_recall': float(test_metrics['recall']),
        'test_f1': float(test_metrics['f1']),
        'test_auc': float(test_metrics['auc']),
        'best_val_acc': float(best_val_acc),
        'optimal_threshold': float(optimal_threshold),
        'total_training_time_minutes': float((phase1_time + phase2_time) / 60)
    }

    with open(Config.OUTPUT_DIR / 'final_results.json', 'w') as f:
        json.dump(final_results, f, indent = 4)

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

    print(f"\n✓ All results saved to {Config.OUTPUT_DIR}")
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE! 🎉")
    print("=" * 80)

# ------------------------------------------------------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize = (8, 6))
    sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', cbar = True, xticklabels = ['Normal', 'AD'], yticklabels = ['Normal', 'AD'])
    plt.xlabel('Predicted', fontsize = 12)
    plt.ylabel('True', fontsize = 12)
    plt.title('Confusion Matrix - Test Set', fontsize = 14, fontweight = 'bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
    plt.close()

# ------------------------------------------------------------------------------------------------------------------
def plot_roc_curve(y_true, y_probs, save_path):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)
    plt.figure(figsize = (8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth = 2, label = f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth = 2, label = 'Random Classifier')
    plt.xlabel('False Positive Rate', fontsize = 12)
    plt.ylabel('True Positive Rate', fontsize = 12)
    plt.title('ROC Curve - Test Set', fontsize = 14, fontweight = 'bold')
    plt.legend()
    plt.grid(True, alpha = 0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
    plt.close()

# ------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    train_model()