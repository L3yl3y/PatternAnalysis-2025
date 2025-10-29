"""
Prediction script for trained GFNet model
Includes test-time augmentation for improved accuracy
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, confusion_matrix, roc_auc_score,
                            classification_report)

from dataset import ADNIDataset, get_transforms
from modules import create_gfnet_model


# ==================================================================================
# CONFIGURATION
# ==================================================================================
class PredictConfig:
    MODEL_PATH = './gfnet_results/checkpoints/best_model.pth'
    DATA_ROOT = './data/ADNI/AD_NC'
    OUTPUT_DIR = Path('./gfnet_results/predictions')
    
    BATCH_SIZE = 32
    IMG_SIZE = 210
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test-time augmentation
    USE_TTA = True
    TTA_SAMPLES = 10
    
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


# ==================================================================================
# TEST-TIME AUGMENTATION
# ==================================================================================
@torch.no_grad()
def test_time_augmentation(model, image_batch, num_augmentations=10):
    """
    Apply test-time augmentation to improve predictions
    Averages predictions over multiple augmented versions
    """
    model.eval()
    device = image_batch.device
    all_predictions = []
    
    # Original prediction
    output = model(image_batch)
    all_predictions.append(F.softmax(output, dim=1))
    
    # Augmented predictions
    for i in range(num_augmentations - 1):
        aug_batch = image_batch.clone()
        
        # Apply different augmentations
        if i % 3 == 0:
            # Horizontal flip
            aug_batch = torch.flip(aug_batch, dims=[3])
        
        if i % 3 == 1:
            # Slight rotation via roll
            shifts = torch.randint(-5, 6, (1,)).item()
            aug_batch = torch.roll(aug_batch, shifts=shifts, dims=2)
        
        if i % 3 == 2:
            # Brightness adjustment
            factor = 0.9 + np.random.random() * 0.2
            aug_batch = torch.clamp(aug_batch * factor, -3, 3)
        
        # Add small noise
        if i % 2 == 0:
            noise = torch.randn_like(aug_batch) * 0.02
            aug_batch = aug_batch + noise
        
        output = model(aug_batch)
        all_predictions.append(F.softmax(output, dim=1))
    
    # Average all predictions
    avg_predictions = torch.stack(all_predictions).mean(dim=0)
    return avg_predictions


# ==================================================================================
# LOAD MODEL
# ==================================================================================
def load_trained_model(model_path, device='cuda'):
    """Load trained GFNet model"""
    print("\n📦 Loading trained model...")
    
    # Create model architecture
    model = create_gfnet_model(
        img_size=210,
        patch_size=14,
        in_chans=1,
        num_classes=2,
        embed_dim=384,
        depth=12,
        drop_rate=0.1,
        drop_path_rate=0.1
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    epoch_info = f" from epoch {checkpoint.get('epoch', 'unknown')}" if 'epoch' in checkpoint else ""
    print(f"   ✅ Model loaded{epoch_info}")
    print(f"   Val Accuracy: {checkpoint.get('val_acc', 'N/A'):.4f}")
    print(f"   Val Recall: {checkpoint.get('val_recall', 'N/A'):.4f}")
    
    return model


# ==================================================================================
# PREDICTION FUNCTIONS
# ==================================================================================
@torch.no_grad()
def predict_dataset(model, dataloader, device='cuda', use_tta=True, tta_samples=10):
    """Predict on entire dataset"""
    model.eval()
    all_results = []
    
    desc = f"Predicting with {'TTA' if use_tta else 'standard'} inference"
    for images, labels in tqdm(dataloader, desc=desc):
        images = images.to(device)
        batch_size = images.size(0)
        
        if use_tta:
            probs = test_time_augmentation(model, images, tta_samples)
        else:
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
        
        # Process each sample
        for i in range(batch_size):
            prob = probs[i]
            pred = torch.argmax(prob).item()
            label = labels[i].item()
            
            all_results.append({
                'true_label': label,
                'prediction': pred,
                'confidence': prob[pred].item(),
                'prob_nc': prob[0].item(),
                'prob_ad': prob[1].item(),
                'correct': label == pred
            })
    
    return all_results


# ==================================================================================
# METRICS & VISUALIZATION
# ==================================================================================
def calculate_metrics(results):
    """Calculate comprehensive metrics"""
    y_true = [r['true_label'] for r in results]
    y_pred = [r['prediction'] for r in results]
    y_probs = [r['prob_ad'] for r in results]
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0
    }
    
    # Per-class metrics
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return metrics, cm


def plot_confusion_matrix(results, save_path):
    """Plot detailed confusion matrix"""
    y_true = [r['true_label'] for r in results]
    y_pred = [r['prediction'] for r in results]
    
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6) * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Confusion Matrix
    labels = [[f"{val}\n({pct:.1f}%)" for val, pct in zip(row_cm, row_pct)] 
              for row_cm, row_pct in zip(cm, cm_percent)]
    
    sns.heatmap(cm, annot=labels, fmt='', cmap='RdYlGn_r', cbar=True,
                xticklabels=['Pred: NC', 'Pred: AD'],
                yticklabels=['True: NC', 'True: AD'],
                ax=ax1, linewidths=2, linecolor='black',
                annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Bar plot
    tn, fp, fn, tp = cm.ravel()
    categories = ['True\nNegative', 'False\nPositive', 'False\nNegative', 'True\nPositive']
    values = [tn, fp, fn, tp]
    colors = ['green', 'orange', 'red', 'darkgreen']
    
    bars = ax2.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Classification Results', fontsize=14, fontweight='bold')
    
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{val}', ha='center', fontweight='bold', fontsize=12)
    
    # Add metrics
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics_text = (f'AD Recall: {recall:.3f}\n'
                   f'AD Precision: {precision:.3f}\n'
                   f'Specificity: {specificity:.3f}')
    ax2.text(0.98, 0.97, metrics_text, transform=ax2.transAxes,
            fontsize=12, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    plt.suptitle('GFNet Test Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confidence_distribution(results, save_path):
    """Plot confidence distribution"""
    correct = [r['confidence'] for r in results if r['correct']]
    incorrect = [r['confidence'] for r in results if not r['correct']]
    
    plt.figure(figsize=(12, 6))
    
    bins = np.linspace(0.5, 1.0, 25)
    plt.hist(correct, bins=bins, alpha=0.7, label=f'Correct (n={len(correct)})',
             color='green', edgecolor='black')
    if incorrect:
        plt.hist(incorrect, bins=bins, alpha=0.7, label=f'Incorrect (n={len(incorrect)})',
                color='red', edgecolor='black')
    
    plt.axvline(np.mean(correct), color='darkgreen', linestyle='--',
               label=f'Mean Correct: {np.mean(correct):.3f}', linewidth=2)
    if incorrect:
        plt.axvline(np.mean(incorrect), color='darkred', linestyle='--',
                   label=f'Mean Incorrect: {np.mean(incorrect):.3f}', linewidth=2)
    
    plt.xlabel('Confidence Score', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ==================================================================================
# MAIN PREDICTION PIPELINE
# ==================================================================================
def run_predictions():
    """Main prediction function"""
    print("\n" + "="*80)
    print("GFNET - ALZHEIMER'S DISEASE PREDICTION")
    print("="*80)
    
    device = torch.device(PredictConfig.DEVICE)
    print(f"\n🖥️ Device: {device}")
    if device.type == 'cuda':
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    model = load_trained_model(PredictConfig.MODEL_PATH, device)
    
    # Load test dataset
    print(f"\n📂 Loading test dataset...")
    test_dataset = ADNIDataset(
        data_root=PredictConfig.DATA_ROOT,
        mode='test',
        transform=get_transforms('test', PredictConfig.IMG_SIZE),
        img_size=PredictConfig.IMG_SIZE,
        use_clahe=False
    )
    
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=PredictConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"   Test samples: {len(test_dataset)}")
    
    # Run predictions
    results = predict_dataset(
        model, test_loader, device,
        use_tta=PredictConfig.USE_TTA,
        tta_samples=PredictConfig.TTA_SAMPLES
    )
    
    # Calculate metrics
    metrics, cm = calculate_metrics(results)
    
    print(f"\n" + "="*80)
    print("🎯 TEST RESULTS:")
    print("="*80)
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  AUC:       {metrics['auc']:.4f}")
    print(f"\n  Per-class:")
    print(f"  • Specificity:      {metrics['specificity']:.4f}")
    print(f"  • Sensitivity (AD): {metrics['sensitivity']:.4f}")
    
    # Error analysis
    errors = [r for r in results if not r['correct']]
    if errors:
        fn = [r for r in errors if r['true_label'] == 1]
        fp = [r for r in errors if r['true_label'] == 0]
        print(f"\n  Errors: {len(errors)}/{len(results)} ({100*len(errors)/len(results):.1f}%)")
        if fn:
            print(f"  • False Negatives (AD→NC): {len(fn)}")
        if fp:
            print(f"  • False Positives (NC→AD): {len(fp)}")
    
    # Save results
    with open(PredictConfig.OUTPUT_DIR / 'test_results.json', 'w') as f:
        json.dump({
            'metrics': {k: float(v) for k, v in metrics.items()},
            'num_samples': len(results),
            'num_correct': sum(1 for r in results if r['correct']),
            'tta_used': PredictConfig.USE_TTA,
            'confusion_matrix': cm.tolist()
        }, f, indent=4)
    
    # Generate visualizations
    print(f"\n📊 Generating visualizations...")
    plot_confusion_matrix(results, PredictConfig.OUTPUT_DIR / 'confusion_matrix.png')
    plot_confidence_distribution(results, PredictConfig.OUTPUT_DIR / 'confidence_dist.png')
    
    print(f"\n✅ Results saved to {PredictConfig.OUTPUT_DIR}")
    
    # Print classification report
    y_true = [r['true_label'] for r in results]
    y_pred = [r['prediction'] for r in results]
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT:")
    print("="*80)
    print(classification_report(y_true, y_pred, target_names=['NC', 'AD']))
    
    if metrics['accuracy'] >= 0.80:
        print("\n" + "="*80)
        print("🎉 SUCCESS! Achieved target accuracy! 🎉")
        print("="*80)
    
    return metrics


if __name__ == '__main__':
    run_predictions()