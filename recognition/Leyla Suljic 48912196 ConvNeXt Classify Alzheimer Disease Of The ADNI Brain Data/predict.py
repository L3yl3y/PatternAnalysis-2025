import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from tqdm import tqdm
from dataset import ADNIDataset, get_transforms
from modules import ConvNeXtAlzheimer

# ------------------------------------------------------------------------------------------------------------------
class PredictConfig:
    pathOfModel = './alzheimer_results_v5_3branch/checkpoints/best_model_final.pth'
    rootOfData = './data/ADNI/AD_NC'
    outputDirectory = Path('./alzheimer_results_v5_3branch/predictions')

    BATCH_SIZE = 8
    IMG_SIZE = 224
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    outputDirectory.mkdir(exist_ok=True, parents=True)

# ------------------------------------------------------------------------------------------------------------------
def load_trained_model(model_path, device='cuda'):
    """Load the trained 3-branch ConvNeXt model"""
    checkpoint = torch.load(model_path, map_location=device)

    # Create model with 3-branch architecture
    model = ConvNeXtAlzheimer(
        model_size='base',
        num_classes=2,
        pretrained=False,
        dropout=0.3
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ 3-Branch Model loaded successfully!")
    print(f"  Architecture: GAP + GMP + Attention-weighted Pooling")
    print(f"  Trained for: {checkpoint['epoch']} epochs")
    print(f"  Val Accuracy: {checkpoint['val_acc']:.4f}")
    print(f"  Val AUC: {checkpoint.get('val_auc', 'N/A')}")

    return model

# ------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def predict_single_image(model, image_tensor, device='cuda'):
    """Predict a single image"""
    model.eval()

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)
    outputs = model(image_tensor)
    probs = F.softmax(outputs, dim=1)
    prediction = torch.argmax(probs, dim=1).item()
    confidence = probs[0, prediction].item()

    probabilities = {
        'Normal': probs[0, 0].item(),
        'AD': probs[0, 1].item()
    }

    return prediction, confidence, probabilities

# ------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def predict_batch(model, dataloader, device='cuda'):
    """Predict a batch of images"""
    model.eval()
    results = []

    for images, labels in tqdm(dataloader, desc='Predicting'):
        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        for i in range(len(images)):
            results.append({
                'true_label': labels[i].item(),
                'prediction': preds[i].item(),
                'confidence': probs[i, preds[i]].item(),
                'prob_normal': probs[i, 0].item(),
                'prob_ad': probs[i, 1].item()
            })

    return results

# ------------------------------------------------------------------------------------------------------------------
def visualise_prediction(image_tensor, true_label, pred_label, probabilities, save_path=None):
    """Visualize a single prediction"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image_tensor.cpu().numpy().transpose(1, 2, 0)
    image = (image * std + mean).clip(0, 1)
    
    axes[0].imshow(image)
    axes[0].axis('off')
    
    true_label_str = 'AD' if true_label == 1 else 'Normal'
    pred_label_str = 'AD' if pred_label == 1 else 'Normal'
    color = 'green' if true_label == pred_label else 'red'
    axes[0].set_title(f'True: {true_label_str} | Pred: {pred_label_str}',
                     fontsize=14, fontweight='bold', color=color)
    
    # Probability bars
    classes = ['Normal', 'AD']
    probs = [probabilities['Normal'], probabilities['AD']]
    colors = ['skyblue', 'salmon']

    bars = axes[1].barh(classes, probs, color=colors, alpha=0.7)
    axes[1].set_xlim([0, 1])
    axes[1].set_xlabel('Probability', fontsize=12)
    axes[1].set_title('Class Probabilities', fontsize=14, fontweight='bold')

    for i, (bar, prob) in enumerate(zip(bars, probs)):
        axes[1].text(prob + 0.02, bar.get_y() + bar.get_height() / 2,
                    f'{prob:.3f}', va='center', fontsize=11, fontweight='bold')

    confidence = max(probs)
    axes[1].text(0.5, -0.15, f'Confidence: {confidence:.1%}',
                ha='center', fontsize=12, fontweight='bold',
                transform=axes[1].transAxes)
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# ------------------------------------------------------------------------------------------------------------------
def visualize_multiple_predictions(model, dataset, num_samples=12, save_path=None):
    """Visualize multiple predictions in a grid"""
    device = PredictConfig.DEVICE
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    rows = 3
    cols = num_samples // rows
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()

    for idx, ax in zip(indices, axes):
        image, true_label = dataset[idx]
        pred, confidence, probs = predict_single_image(model, image, device)
        
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_display = image.cpu().numpy().transpose(1, 2, 0)
        img_display = (img_display * std + mean).clip(0, 1)

        ax.imshow(img_display)
        ax.axis('off')
        
        true_str = 'AD' if true_label == 1 else 'Normal'
        pred_str = 'AD' if pred == 1 else 'Normal'
        color = 'green' if pred == true_label else 'red'

        ax.set_title(f'True: {true_str} | Pred: {pred_str}\nConf: {confidence:.2f}',
                    fontsize=10, color=color, fontweight='bold')

    plt.suptitle('Sample Predictions (3-Branch Model)', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# ------------------------------------------------------------------------------------------------------------------
def plot_confidence_distribution(results, save_path=None):
    """Plot confidence distribution for correct vs incorrect predictions"""
    correct_conf = [r['confidence'] for r in results if r['true_label'] == r['prediction']]
    wrong_conf = [r['confidence'] for r in results if r['true_label'] != r['prediction']]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bins = np.linspace(0, 1, 20)
    axes[0].hist(correct_conf, bins=bins, alpha=0.7, label='Correct',
                color='green', edgecolor='black')
    axes[0].hist(wrong_conf, bins=bins, alpha=0.7, label='Incorrect',
                color='red', edgecolor='black')
    axes[0].set_xlabel('Confidence', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Confidence Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    data = [correct_conf, wrong_conf]
    bp = axes[1].boxplot(data, labels=['Correct', 'Incorrect'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    axes[1].set_ylabel('Confidence', fontsize=12)
    axes[1].set_title('Confidence Comparison', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# ------------------------------------------------------------------------------------------------------------------
def plot_error_analysis(results, save_path=None):
    """Analyze errors and create visualizations"""
    correct = sum(1 for r in results if r['true_label'] == r['prediction'])
    total = len(results)

    true_pos = sum(1 for r in results if r['true_label'] == 1 and r['prediction'] == 1)
    false_pos = sum(1 for r in results if r['true_label'] == 0 and r['prediction'] == 1)
    false_neg = sum(1 for r in results if r['true_label'] == 1 and r['prediction'] == 0)
    true_neg = sum(1 for r in results if r['true_label'] == 0 and r['prediction'] == 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pie chart
    labels = ['Correct', 'Incorrect']
    sizes = [correct, total - correct]
    colors = ['lightgreen', 'lightcoral']
    explode = (0.05, 0)

    axes[0].pie(sizes, explode=explode, labels=labels, colors=colors,
               autopct='%1.1f%%', startangle=90,
               textprops={'fontsize': 12, 'fontweight': 'bold'})
    axes[0].set_title(f'Overall Accuracy: {correct / total:.1%}',
                     fontsize=14, fontweight='bold')
    
    # Bar chart
    error_types = ['True Pos', 'False Pos', 'False Neg', 'True Neg']
    counts = [true_pos, false_pos, false_neg, true_neg]
    colors = ['green', 'red', 'orange', 'green']

    axes[1].bar(error_types, counts, color=colors, alpha=0.7)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Error Type Analysis', fontsize=14, fontweight='bold')

    for i, count in enumerate(counts):
        axes[1].text(i, count + 0.5, str(count), ha='center',
                    fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# ------------------------------------------------------------------------------------------------------------------
def calculate_metrics(results):
    """Calculate comprehensive metrics"""
    y_true = [r['true_label'] for r in results]
    y_pred = [r['prediction'] for r in results]
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    probs = [r['prob_ad'] for r in results]
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.0
    }
    
    return metrics

# ------------------------------------------------------------------------------------------------------------------
def run_predictions():
    """Main prediction pipeline"""
    print("=" * 80)
    print("PREDICTION PIPELINE - 3-BRANCH CONVNEXT MODEL")
    print("=" * 80)
    
    model = load_trained_model(PredictConfig.pathOfModel, PredictConfig.DEVICE)

    test_dataset = ADNIDataset(
        data_root=PredictConfig.rootOfData,
        mode='test',
        transform=get_transforms('test', PredictConfig.IMG_SIZE),
        img_size=PredictConfig.IMG_SIZE,
        use_clahe=False  # No CLAHE for test
    )

    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=PredictConfig.BATCH_SIZE,
                            shuffle=False, num_workers=8)
    
    print(f"\nRunning predictions on {len(test_dataset)} test images...")
    results = predict_batch(model, test_loader, PredictConfig.DEVICE)

    # Calculate metrics
    metrics = calculate_metrics(results)
    
    print(f"\n{'='*80}")
    print("TEST SET METRICS:")
    print(f"{'='*80}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  AUC:       {metrics['auc']:.4f}")
    print(f"{'='*80}")

    # Save metrics
    with open(PredictConfig.outputDirectory / 'test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    # PLOT 1: Prediction grid
    print("\nGenerating visualizations...")
    visualize_multiple_predictions(model, test_dataset, num_samples=12,
                                   save_path=PredictConfig.outputDirectory / 'prediction_grid.png')

    # PLOT 2: Confidence distribution
    plot_confidence_distribution(results,
                                save_path=PredictConfig.outputDirectory / 'confidence_distribution.png')

    # PLOT 3: Error analysis
    plot_error_analysis(results,
                       save_path=PredictConfig.outputDirectory / 'error_analysis.png')

    # PLOT 4: Individual examples
    # Correct Normal
    correct_normal = next((i, r) for i, r in enumerate(results)
                         if r['true_label'] == 0 and r['prediction'] == 0)
    image, label = test_dataset[correct_normal[0]]
    pred, conf, probs = predict_single_image(model, image, PredictConfig.DEVICE)
    visualise_prediction(image, label, pred, probs,
                        save_path=PredictConfig.outputDirectory / 'example_correct_normal.png')

    # Correct AD
    correct_ad = next((i, r) for i, r in enumerate(results)
                     if r['true_label'] == 1 and r['prediction'] == 1)
    image, label = test_dataset[correct_ad[0]]
    pred, conf, probs = predict_single_image(model, image, PredictConfig.DEVICE)
    visualise_prediction(image, label, pred, probs,
                        save_path=PredictConfig.outputDirectory / 'example_correct_ad.png')

    # Wrong prediction (if any)
    wrong_examples = [(i, r) for i, r in enumerate(results)
                     if r['true_label'] != r['prediction']]
    
    if wrong_examples:
        image, label = test_dataset[wrong_examples[0][0]]
        pred, conf, probs = predict_single_image(model, image, PredictConfig.DEVICE)
        visualise_prediction(image, label, pred, probs,
                            save_path=PredictConfig.outputDirectory / 'example_wrong_prediction.png')
        print(f"\nFound {len(wrong_examples)} wrong predictions")
    else:
        print("\n✅ Perfect predictions! No errors found!")

    print(f"\n✅ All visualizations saved to {PredictConfig.outputDirectory}")
    
    # Final summary
    if metrics['accuracy'] >= 0.8 and metrics['recall'] >= 0.8:
        print("\n" + "="*80)
        print("🎉 TARGET METRICS ACHIEVED! 🎉")
        print("="*80)
    else:
        print("\n⚠️  Some targets not met:")
        if metrics['accuracy'] < 0.8:
            print(f"   Accuracy: {metrics['accuracy']:.4f} (target: 0.8)")
        if metrics['recall'] < 0.8:
            print(f"   Recall: {metrics['recall']:.4f} (target: 0.8)")

# ------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    run_predictions()