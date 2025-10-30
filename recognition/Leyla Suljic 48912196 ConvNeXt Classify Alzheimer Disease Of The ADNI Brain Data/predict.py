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
# UPDATED FOR V3 MODEL
# ------------------------------------------------------------------------------------------------------------------
class PredictConfig:
    # Updated path for v3 model
    pathOfModel = './alzheimer_results_v3/checkpoints/best_model.pth'
    rootOfData = './data/ADNI/AD_NC'
    outputDirectory = Path('./alzheimer_results_v3/predictions')

    BATCH_SIZE = 1
    IMG_SIZE = 224
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    outputDirectory.mkdir(exist_ok = True, parents = True)

# ------------------------------------------------------------------------------------------------------------------
def load_trained_model(model_path, device='cuda'):
    checkpoint = torch.load(model_path, map_location = device)

    # UPDATED: Using 'small' model with dropout=0.35
    model = ConvNeXtAlzheimer(
        model_size = 'small',  # Updated from 'tiny'
        num_classes = 2,
        pretrained = False,
        dropout = 0.35  # Updated to match v3 config
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✔ Model loaded successfully!")
    print(f"  Model: ConvNeXt-small")
    print(f"  Trained for: {checkpoint['epoch']} epochs")
    
    # Display saved metrics
    if 'test_acc' in checkpoint:
        print(f"  Best Test Accuracy: {checkpoint['test_acc']:.4f}")
    if 'test_recall' in checkpoint:
        print(f"  Best Test Recall: {checkpoint['test_recall']:.4f}")
    if 'threshold' in checkpoint:
        print(f"  Decision Threshold: {checkpoint['threshold']:.3f}")
        return model, checkpoint['threshold']

    return model, 0.5  # Default threshold

# ------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def predict_single_image(model, image_tensor, device = 'cuda', threshold=0.5):
    model.eval()

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)
    outputs = model(image_tensor)
    probs = F.softmax(outputs, dim = 1)
    
    # Use custom threshold
    ad_prob = probs[0, 1].item()
    prediction = 1 if ad_prob >= threshold else 0
    confidence = probs[0, prediction].item()

    probabilities = {
        'Normal': probs[0, 0].item(),
        'AD': probs[0, 1].item()
    }

    return prediction, confidence, probabilities

# ------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def predict_batch(model, dataloader, device='cuda', threshold=0.5):
    model.eval()
    results = []

    for images, labels in tqdm(dataloader, desc='Making predictions'):
        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim = 1)
        
        # Use custom threshold
        ad_probs = probs[:, 1]
        preds = (ad_probs >= threshold).long()

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
def visualise_prediction(image_tensor, true_label, pred_label, probabilities, save_path = None):
    fig, axes = plt.subplots(1, 2, figsize = (12, 5))
    
    # Display image
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
                     fontsize = 14, fontweight = 'bold', color = color)
    
    # Display probabilities
    classes = ['Normal', 'AD']
    probs = [probabilities['Normal'], probabilities['AD']]
    colors = ['skyblue', 'salmon']

    bars = axes[1].barh(classes, probs, color = colors, alpha = 0.7)
    axes[1].set_xlim([0, 1])
    axes[1].set_xlabel('Probability', fontsize = 12)
    axes[1].set_title('Class Probabilities', fontsize = 14, fontweight = 'bold')

    for i, (bar, prob) in enumerate(zip(bars, probs)):
        axes[1].text(prob + 0.02, bar.get_y() + bar.get_height() / 2, 
                    f'{prob:.3f}', va = 'center', fontsize = 11, fontweight = 'bold')

    confidence = max(probs)
    axes[1].text(0.5, -0.15, f'Confidence: {confidence:.1%}', 
                ha = 'center', fontsize = 12, fontweight = 'bold', 
                transform = axes[1].transAxes)
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
        plt.close()
    else:
        plt.show()

# ------------------------------------------------------------------------------------------------------------------
def plot_confidence_distribution(results, save_path = None):
    correct_conf = [r['confidence'] for r in results if r['true_label'] == r['prediction']]
    wrong_conf = [r['confidence'] for r in results if r['true_label'] != r['prediction']]
    
    fig, axes = plt.subplots(1, 2, figsize = (14, 5))

    # Histogram
    bins = np.linspace(0, 1, 20)
    axes[0].hist(correct_conf, bins = bins, alpha = 0.7, 
                 label = f'Correct (n={len(correct_conf)})', color = 'green', edgecolor = 'black')
    axes[0].hist(wrong_conf, bins = bins, alpha = 0.7, 
                 label = f'Incorrect (n={len(wrong_conf)})', color = 'red', edgecolor = 'black')
    axes[0].set_xlabel('Confidence', fontsize = 12)
    axes[0].set_ylabel('Count', fontsize = 12)
    axes[0].set_title('Confidence Distribution', fontsize = 14, fontweight = 'bold')
    axes[0].legend()
    axes[0].grid(True, alpha = 0.3)

    # Boxplot
    data = [correct_conf, wrong_conf]
    bp = axes[1].boxplot(data, labels = ['Correct', 'Incorrect'], patch_artist = True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    axes[1].set_ylabel('Confidence', fontsize = 12)
    axes[1].set_title('Confidence Comparison', fontsize = 14, fontweight = 'bold')
    axes[1].grid(True, alpha = 0.3, axis = 'y')
    
    # Add mean values
    for i, d in enumerate(data):
        if len(d) > 0:
            mean_val = np.mean(d)
            axes[1].text(i+1, mean_val, f'μ={mean_val:.3f}', 
                        ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
        plt.close()
    else:
        plt.show()

# ------------------------------------------------------------------------------------------------------------------
def plot_error_analysis(results, save_path = None):
    correct = sum(1 for r in results if r['true_label'] == r['prediction'])
    total = len(results)

    true_pos = sum(1 for r in results if r['true_label'] == 1 and r['prediction'] == 1)
    false_pos = sum(1 for r in results if r['true_label'] == 0 and r['prediction'] == 1)
    false_neg = sum(1 for r in results if r['true_label'] == 1 and r['prediction'] == 0)
    true_neg = sum(1 for r in results if r['true_label'] == 0 and r['prediction'] == 0)

    fig, axes = plt.subplots(1, 3, figsize = (18, 5))
    
    # Pie chart
    labels = ['Correct', 'Incorrect']
    sizes = [correct, total - correct]
    colors = ['lightgreen', 'lightcoral']
    explode = (0.05, 0)

    axes[0].pie(sizes, explode = explode, labels = labels, colors = colors, 
                autopct = '%1.1f%%', startangle = 90, 
                textprops = {'fontsize': 12, 'fontweight': 'bold'})
    axes[0].set_title(f'Overall Accuracy: {correct / total:.1%}', 
                     fontsize = 14, fontweight = 'bold')
    
    # Error types bar chart
    error_types = ['True Pos', 'False Pos', 'False Neg', 'True Neg']
    counts = [true_pos, false_pos, false_neg, true_neg]
    colors = ['green', 'red', 'orange', 'blue']

    axes[1].bar(error_types, counts, color = colors, alpha = 0.7)
    axes[1].set_ylabel('Count', fontsize = 12)
    axes[1].set_title('Classification Breakdown', fontsize = 14, fontweight = 'bold')

    for i, count in enumerate(counts):
        axes[1].text(i, count + 0.5, str(count), ha = 'center', 
                    fontsize = 12, fontweight = 'bold')

    # Confusion Matrix
    cm = [[true_neg, false_pos], [false_neg, true_pos]]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
                xticklabels=['Normal', 'AD'], yticklabels=['Normal', 'AD'])
    axes[2].set_xlabel('Predicted', fontsize = 12)
    axes[2].set_ylabel('Actual', fontsize = 12)
    axes[2].set_title('Confusion Matrix', fontsize = 14, fontweight = 'bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
        plt.close()
    else:
        plt.show()

# ------------------------------------------------------------------------------------------------------------------
def run_predictions():
    print("=" * 80)
    print("RUNNING PREDICTIONS WITH V3 OPTIMIZED MODEL")
    print("=" * 80)
    
    model, optimal_threshold = load_trained_model(PredictConfig.pathOfModel, PredictConfig.DEVICE)
    print(f"  Using threshold: {optimal_threshold:.3f}")

    test_dataset = ADNIDataset(
        data_root = PredictConfig.rootOfData,
        mode = 'test',
        transform = get_transforms('test', PredictConfig.IMG_SIZE),
        img_size = PredictConfig.IMG_SIZE
    )

    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size = 8, shuffle = False, num_workers = 8)
    
    print("\n📊 Evaluating on test set...")
    results = predict_batch(model, test_loader, PredictConfig.DEVICE, threshold=optimal_threshold)

    # Calculate metrics
    correct = sum(1 for r in results if r['true_label'] == r['prediction'])
    total = len(results)
    accuracy = correct / total
    
    true_pos = sum(1 for r in results if r['true_label'] == 1 and r['prediction'] == 1)
    false_pos = sum(1 for r in results if r['true_label'] == 0 and r['prediction'] == 1)
    false_neg = sum(1 for r in results if r['true_label'] == 1 and r['prediction'] == 0)
    true_neg = sum(1 for r in results if r['true_label'] == 0 and r['prediction'] == 0)
    
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    specificity = true_neg / (true_neg + false_pos) if (true_neg + false_pos) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Generate visualizations
    print("\n📊 Generating visualization plots...")
    plot_confidence_distribution(results, save_path = PredictConfig.outputDirectory / 'confidence_distribution.png')
    plot_error_analysis(results, save_path = PredictConfig.outputDirectory / 'error_analysis.png')

    # Print detailed results
    print("\n" + "=" * 80)
    print("🎯 TEST SET RESULTS (V3 OPTIMIZED MODEL)")
    print("=" * 80)
    print(f"  Total samples: {total}")
    print(f"  Correct predictions: {correct}")
    print(f"  Incorrect predictions: {total - correct}")
    print("\n📊 Performance Metrics:")
    print(f"  Accuracy:    {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"  Recall:      {recall:.4f} ({recall * 100:.2f}%)")
    print(f"  Precision:   {precision:.4f} ({precision * 100:.2f}%)")
    print(f"  Specificity: {specificity:.4f} ({specificity * 100:.2f}%)")
    print(f"  F1 Score:    {f1:.4f} ({f1 * 100:.2f}%)")
    print("\n📊 Confusion Matrix:")
    print(f"  True Positives:  {true_pos} (Correctly identified AD)")
    print(f"  True Negatives:  {true_neg} (Correctly identified Normal)")
    print(f"  False Positives: {false_pos} (Normal misclassified as AD)")
    print(f"  False Negatives: {false_neg} (AD misclassified as Normal)")
    print("=" * 80)
    
    # Target achievement check
    if accuracy >= 0.8 and recall >= 0.8:
        print("✅ BOTH TARGETS ACHIEVED! (Accuracy & Recall ≥ 80%)")
    elif accuracy >= 0.8:
        print("⚠️  Accuracy target met, but recall below 80%")
    elif recall >= 0.8:
        print("⚠️  Recall target met, but accuracy below 80%")
    else:
        print("❌ Below target (need both Accuracy & Recall ≥ 80%)")
    
    print(f"\n✅ Visualizations saved to {PredictConfig.outputDirectory}")

# ------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    run_predictions()