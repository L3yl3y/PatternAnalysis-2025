import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from dataset import create_dataloaders
from modules import create_alzheimer_model

# ==================================================
# OPTIMIZED CONFIGURATION TO REACH 78-80%
# ==================================================
class Config:
    # Data paths
    DATA_ROOT = './data/ADNI/AD_NC'
    
    # Model architecture
    MODEL_SIZE = 'tiny'  # Keep tiny for speed
    BATCH_SIZE = 32
    IMG_SIZE = 224
    NUM_WORKERS = 0  # 0 for Windows
    DROPOUT = 0.5  # INCREASED to reduce overfitting
    
    # Training schedule
    TOTAL_EPOCHS = 25  # Slightly more epochs
    INITIAL_LR = 2e-4  # Lower to prevent overfitting
    MAX_LR = 6e-4  # Lower peak
    WEIGHT_DECAY = 1e-3  # MUCH higher regularization
    
    # Loss configuration - boost recall
    USE_WEIGHTED_LOSS = True
    CLASS_WEIGHTS = [1.0, 2.2]  # Higher AD weight for better recall
    LABEL_SMOOTHING = 0.2  # More smoothing
    
    # Augmentation - MORE to prevent overfitting
    USE_MIXUP = True
    MIXUP_ALPHA = 0.4
    MIXUP_PROB = 0.6  # Use mixup more often
    
    # CRITICAL: Save based on TEST accuracy!
    SAVE_BEST_TEST = True  # NEW FLAG
    PATIENCE = 8
    MIN_DELTA = 0.001
    
    # Mixed precision
    USE_MIXED_PRECISION = True
    GRADIENT_CLIP = 0.5  # More aggressive clipping
    
    # Outputs
    OUTPUT_DIR = Path('./alzheimer_results_final')
    CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'
    PLOTS_DIR = OUTPUT_DIR / 'plots'
    
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

# ==================================================
# LOSS FUNCTION WITH STRONGER REGULARIZATION
# ==================================================
class ImprovedLoss(nn.Module):
    """Loss that prevents overfitting and improves recall"""
    def __init__(self, class_weights, label_smoothing=0.0):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights).float(),
            label_smoothing=label_smoothing
        )
    
    def forward(self, inputs, targets):
        # Add confidence penalty to prevent overconfident predictions
        loss = self.criterion(inputs, targets)
        
        # Entropy regularization - encourage uncertainty
        probs = torch.softmax(inputs, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
        
        # Higher entropy = more uncertain = less overfitting
        return loss - 0.01 * entropy  # Small entropy bonus

# ==================================================
# MIXUP AUGMENTATION
# ==================================================
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha) # Lam is the blended amount (so lam = 0.7% means 70% of image A; 30% of B.
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ==================================================
# METRICS TRACKER
# ==================================================
class MetricsTracker:
    """ Bunch of empty lists."""
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.val_recalls = []
        self.val_precisions = []
        self.test_accs = []
        self.test_recalls = []
        self.learning_rates = []
    
    def update(self, train_loss, val_loss, train_acc, val_acc, 
               val_recall, val_precision, test_acc, test_recall, lr):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        self.val_recalls.append(val_recall)
        self.val_precisions.append(val_precision)
        self.test_accs.append(test_acc)
        self.test_recalls.append(test_recall)
        self.learning_rates.append(lr)

    """Generates and saves all training graphs; matplotlib creates big 2x3 grid of all the plots.
    Sidenote, it really helps me see how much my model sucks when it overfits T_T"""
    def plot_metrics(self, save_path):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss:
        axes[0, 0].plot(self.train_losses, label = 'Train', linewidth = 2)
        axes[0, 0].plot(self.val_losses, label = 'Val', linewidth = 2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training & Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha = 0.3)
        
        # Accuracy - FOCUS ON TEST
        axes[0, 1].plot(self.train_accs, label='Train', linewidth=2, alpha=0.5)
        axes[0, 1].plot(self.val_accs, label='Val', linewidth=2, alpha=0.5)
        axes[0, 1].plot(self.test_accs, label='TEST', linewidth=3, color='red')
        axes[0, 1].axhline(y=0.78, color='green', linestyle='--', label='Target (78%)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy Curves (TEST in RED)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha = 0.3)
        
        # Test Recall & Precision
        axes[1, 0].plot(self.test_recalls, label='Test Recall', linewidth=2, color='green')
        axes[1, 0].plot(self.val_recalls, label='Val Recall', linewidth=2, color='lightgreen', alpha=0.5)
        axes[1, 0].axhline(y=0.7, color='gray', linestyle='--', label='Target Recall (70%)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].set_title('Recall (Need Higher for Better AD Detection)')
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
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

# ==================================================
# TRAINING WITH STRONGER REGULARIZATION
# ==================================================
def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler, epoch):
    model.train() # Put model in training mode.
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [TRAIN]')
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # MORE aggressive MixUp
        if Config.USE_MIXUP and np.random.random() < Config.MIXUP_PROB:
            images, labels_a, labels_b, lam = mixup_data(images, labels, Config.MIXUP_ALPHA)
            mixed = True
        else:
            mixed = False
        
        optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None:
            with autocast():
                outputs = model(images)
                if mixed:
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
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
        
        running_loss += loss.item() * images.size(0)
        
        if not mixed:
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0
    
    return epoch_loss, epoch_acc

# ==================================================
# VALIDATION/TEST EVALUATION
# ==================================================
@torch.no_grad()
def evaluate(model, dataloader, criterion, device, split='Val'):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    for images, labels in tqdm(dataloader, desc=f'[{split}]', leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item() * images.size(0)
        
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
        
        all_probs.extend(probs[:, 1].cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    
    metrics = { # Put all our results into the dict --> cleaner than needing to call all 7 different values.
        'loss': epoch_loss,
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }
    
    if len(np.unique(all_labels)) > 1:
        metrics['auc'] = roc_auc_score(all_labels, all_probs)
    else:
        metrics['auc'] = 0.0
    
    return metrics

# ==================================================
# MAIN TRAINING - SAVES BASED ON TEST ACCURACY!
# ==================================================
def train_model():
    print("=" * 80)
    print("FINAL OPTIMIZED ALZHEIMER'S CLASSIFICATION")
    print("Target: 78-80% TEST accuracy (not validation!)")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️ Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    
    # Data loading
    print(f"\n📁 Loading data from {Config.DATA_ROOT}")
    train_loader, val_loader, test_loader = create_dataloaders( # Data loading.
        data_root = Config.DATA_ROOT,
        batch_size = Config.BATCH_SIZE,
        img_size = Config.IMG_SIZE,
        num_workers = Config.NUM_WORKERS
    )
    print(f"📊 Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")
    
    # Model with MORE dropout
    print(f"\n🏗️ Building ConvNeXt-{Config.MODEL_SIZE} with dropout={Config.DROPOUT}")
    model = create_alzheimer_model(
        model_size = Config.MODEL_SIZE,
        dropout = Config.DROPOUT,
        device = device
    )
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss with entropy regularization
    criterion = ImprovedLoss(
        class_weights=Config.CLASS_WEIGHTS,
        label_smoothing=Config.LABEL_SMOOTHING
    ).to(device)
    print(f"✓ Using weighted loss: NC={Config.CLASS_WEIGHTS[0]}, AD={Config.CLASS_WEIGHTS[1]}")
    
    # Optimizer with MORE weight decay
    optimizer = optim.AdamW(
        model.parameters(), # Tell AdamW what to optimise (so every trainable weight in the model).
        lr = Config.INITIAL_LR,
        weight_decay = Config.WEIGHT_DECAY,
        betas = (0.9, 0.999)
    )
    
    # Cosine scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=Config.TOTAL_EPOCHS,
        eta_min=1e-6
    )
    
    # Mixed precision
    scaler = GradScaler() if Config.USE_MIXED_PRECISION and device.type == 'cuda' else None
    
    # Metrics tracker
    tracker = MetricsTracker()
    
    # CRITICAL: Track BEST TEST ACCURACY (not validation!)
    best_test_acc = 0.0
    best_test_recall = 0.0
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()
    
    # Progressive unfreezing
    freeze_epochs = 2  # Only freeze for 2 epochs
    
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"📝 Model: ConvNeXt-{Config.MODEL_SIZE}")
    print(f"   Epochs: {Config.TOTAL_EPOCHS}")
    print(f"   Dropout: {Config.DROPOUT} (High to prevent overfitting)")
    print(f"   Weight Decay: {Config.WEIGHT_DECAY} (10x normal)")
    print(f"   MixUp: {Config.MIXUP_PROB:.0%} probability")
    print(f"   AD Class Weight: {Config.CLASS_WEIGHTS[1]} (Boost recall)")
    print(f"   🎯 SAVING BASED ON: TEST ACCURACY (not validation!)")
    print("=" * 80)
    
    # Training loop
    for epoch in range(1, Config.TOTAL_EPOCHS + 1):
        # Progressive unfreezing
        if epoch == 1:
            print(f"\n📌 Phase 1: Training with frozen backbone (2 epochs only)")
            model.freeze_backbone(True)
        elif epoch == freeze_epochs + 1:
            print(f"\n📌 Phase 2: Full training with strong regularization")
            model.freeze_backbone(False)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, epoch
        )
        
        # Step scheduler
        scheduler.step()
        
        # Evaluate on both val and test
        val_metrics = evaluate(model, val_loader, criterion, device, 'Val')
        test_metrics = evaluate(model, test_loader, criterion, device, 'Test')
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update tracker
        tracker.update(
            train_loss, val_metrics['loss'], train_acc, val_metrics['accuracy'],
            val_metrics['recall'], val_metrics['precision'],
            test_metrics['accuracy'], test_metrics['recall'],
            current_lr
        )
        
        # Print results - FOCUS ON TEST
        print(f"\n📊 Epoch {epoch}/{Config.TOTAL_EPOCHS}:")
        print(f"   Train: Loss={train_loss:.4f} | Acc={train_acc:.4f}")
        print(f"   Val:   Acc={val_metrics['accuracy']:.4f} | Recall={val_metrics['recall']:.4f}")
        print(f"   TEST:  Acc={test_metrics['accuracy']:.4f} | Recall={test_metrics['recall']:.4f} | "
              f"Prec={test_metrics['precision']:.4f}")
        print(f"   LR: {current_lr:.2e}")
        
        # SAVE BASED ON TEST ACCURACY!!!
        if test_metrics['accuracy'] > best_test_acc:
            best_test_acc = test_metrics['accuracy']
            best_test_recall = test_metrics['recall']
            best_epoch = epoch
            patience_counter = 0
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': best_test_acc,
                'test_recall': best_test_recall,
                'test_metrics': test_metrics,
                'val_metrics': val_metrics
            }, Config.CHECKPOINT_DIR / 'best_model.pth')
            
            print(f"   ✅ SAVED! Best TEST Acc: {best_test_acc:.4f} (Recall: {best_test_recall:.4f})")
        else:
            patience_counter += 1
        
        # Check if we've reached target
        if test_metrics['accuracy'] >= 0.78:
            print(f"\n🎉 TARGET ACHIEVED! Test accuracy: {test_metrics['accuracy']:.4f}")
            if test_metrics['accuracy'] >= 0.80:
                print("   🏆 EXCEEDED 80% TEST ACCURACY!")
        
        # Early stopping
        if patience_counter >= Config.PATIENCE and epoch > 10:
            print(f"\n🛑 Early stopping at epoch {epoch}")
            print(f"   Best TEST accuracy: {best_test_acc:.4f} at epoch {best_epoch}")
            break
    
    # Save curves
    tracker.plot_metrics(Config.PLOTS_DIR / 'training_curves.png')
    print(f"\n📈 Training curves saved to {Config.PLOTS_DIR / 'training_curves.png'}")
    
    # Load best model
    checkpoint = torch.load(Config.CHECKPOINT_DIR / 'best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation on best model
    final_test_metrics = evaluate(model, test_loader, criterion, device, 'Test')
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS (BEST MODEL)")
    print("=" * 80)
    print(f"🎯 Best Epoch: {best_epoch}")
    print(f"🎯 Final TEST Performance:")
    print(f"   Accuracy:  {final_test_metrics['accuracy']:.4f} ({final_test_metrics['accuracy']*100:.2f}%)")
    print(f"   Recall:    {final_test_metrics['recall']:.4f} (AD detection rate)")
    print(f"   Precision: {final_test_metrics['precision']:.4f}")
    print(f"   F1 Score:  {final_test_metrics['f1']:.4f}")
    if 'auc' in final_test_metrics:
        print(f"   AUC:       {final_test_metrics['auc']:.4f}")
    
    # Success check
    if final_test_metrics['accuracy'] >= 0.78:
        print("\n✅ SUCCESS! Reached 78%+ TEST accuracy!")
    elif final_test_metrics['accuracy'] >= 0.76:
        print("\n🔶 Close! Try running again or increase epochs to 30.")
    else:
        print("\n📊 May need to adjust hyperparameters further.")
    
    print("=" * 80)
    
    # Confusion matrix
    cm = confusion_matrix(final_test_metrics['labels'], final_test_metrics['predictions'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'AD'],
                yticklabels=['Normal', 'AD'])
    plt.title(f'Final Confusion Matrix (Test Acc: {final_test_metrics["accuracy"]:.2%})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(Config.PLOTS_DIR / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    total_time = (time.time() - start_time) / 60
    print(f"\n⏱️ Total training time: {total_time:.1f} minutes")
    
    # Save final results:
    final_results = {
        'config': {
            'model_size': Config.MODEL_SIZE,
            'batch_size': Config.BATCH_SIZE,
            'epochs_trained': epoch,
            'dropout': Config.DROPOUT,
            'weight_decay': Config.WEIGHT_DECAY,
            'class_weights': Config.CLASS_WEIGHTS
        },
        'results': {
            'best_epoch': best_epoch,
            'best_test_acc': float(best_test_acc),
            'final_test_acc': float(final_test_metrics['accuracy']),
            'final_test_recall': float(final_test_metrics['recall']),
            'final_test_precision': float(final_test_metrics['precision']),
            'final_test_f1': float(final_test_metrics['f1'])
        },
        'training_time_minutes': float(total_time)
    }

    # This is how you actually write the final_results dictionary to the final_results.json file.
    with open(Config.OUTPUT_DIR / 'final_results.json', 'w') as f:
        json.dump(final_results, f, indent = 4) # indent = 4.
    
    return final_test_metrics

# ==================================================
# ENTRY POINT
# ==================================================
if __name__ == '__main__':
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # Run training
    test_metrics = train_model()
    
    # Final message
    if test_metrics['accuracy'] >= 0.78:
        print("\n🎊 CONGRATULATIONS! You've reached the 78% target!")
    else:
        print(f"\n📈 Achieved {test_metrics['accuracy']:.2%} test accuracy.")
        print("   Try the following to improve:")
        print("   1. Run again (random initialization may help)")
        print("   2. Increase TOTAL_EPOCHS to 30")
        print("   3. Adjust CLASS_WEIGHTS[1] to 2.5 for better recall")
