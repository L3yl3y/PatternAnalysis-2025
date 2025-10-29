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
# FIXED: Updated paths and dropout to match new training script
# ------------------------------------------------------------------------------------------------------------------
class PredictConfig:
    # FIXED: Updated path to match train_fixed.py output
    pathOfModel = './alzheimer_results_fixed/checkpoints/best_model.pth'
    rootOfData = './data/ADNI/AD_NC'
    outputDirectory = Path('./alzheimer_results_fixed/predictions')

    BATCH_SIZE = 1
    IMG_SIZE = 224
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    outputDirectory.mkdir(exist_ok = True, parents = True)

# ------------------------------------------------------------------------------------------------------------------
def load_trained_model(model_path, device='cuda'):
    checkpoint = torch.load(model_path, map_location = device)

    # FIXED: Dropout matches train_fixed.py (0.3)
    model = ConvNeXtAlzheimer(
        model_size = 'base',
        num_classes = 2,
        pretrained = False,
        dropout = 0.3  # Matches training
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully!")
    print(f"  Trained for: {checkpoint['epoch']} epochs")
    
    # FIXED: Handle different checkpoint formats
    if 'val_acc' in checkpoint:
        print(f"  Val Accuracy: {checkpoint['val_acc']:.4f}")
    if 'balanced_metric' in checkpoint:
        print(f"  Balanced Metric: {checkpoint['balanced_metric']:.4f}")

    return model

# ------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def predict_single_image(model, image_tensor, device = 'cuda'):
    model.eval()

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)
    outputs = model(image_tensor)
    probs = F.softmax(outputs, dim = 1)
    prediction = torch.argmax(probs, dim = 1).item()
    confidence = probs[0, prediction].item()

    probabilities = {
        'Normal': probs[0, 0].item(),
        'AD': probs[0, 1].item()
    }

    return prediction, confidence, probabilities

# ------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def predict_batch(model, dataloader, device='cuda'):
    model.eval()
    results = []

    for images, labels in tqdm(dataloader, desc='Making predictions'):
        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim = 1)
        preds = torch.argmax(probs, dim = 1)

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
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image_tensor.cpu().numpy().transpose(1, 2, 0)
    image = (image * std + mean).clip(0, 1)
    axes[0].imshow(image)

    axes[0].axis('off')
    true_label_str = 'AD' if true_label == 1 else 'Normal'
    pred_label_str = 'AD' if pred_label == 1 else 'Normal'
    color = 'green' if true_label == pred_label else 'red'
    axes[0].set_title(f'True: {true_label_str} | Pred: {pred_label_str}', fontsize = 14, fontweight = 'bold', color = color)
    classes = ['Normal', 'AD']
    probs = [probabilities['Normal'], probabilities['AD']]
    colors = ['skyblue', 'salmon']

    bars = axes[1].barh(classes, probs, color = colors, alpha = 0.7)
    axes[1].set_xlim([0, 1])
    axes[1].set_xlabel('Probability', fontsize = 12)
    axes[1].set_title('Class Probabilities', fontsize = 14, fontweight = 'bold')

    for i, (bar, prob) in enumerate(zip(bars, probs)):
        axes[1].text(prob + 0.02, bar.get_y() + bar.get_height() / 2, f'{prob:.3f}', va = 'center', fontsize = 11, fontweight = 'bold')

    confidence = max(probs)
    axes[1].text(0.5, -0.15, f'Confidence: {confidence:.1%}', ha = 'center', fontsize = 12, fontweight = 'bold', transform = axes[1].transAxes)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
        plt.close()
    else:
        plt.show()

# ------------------------------------------------------------------------------------------------------------------
def visualize_multiple_predictions(model, dataset, num_samples = 12, save_path = None):
    device = PredictConfig.DEVICE
    indices = np.random.choice(len(dataset), num_samples, replace = False)
    rows = 3
    cols = num_samples // rows
    fig, axes = plt.subplots(rows, cols, figsize = (cols * 4, rows * 4))
    axes = axes.flatten()

    for idx, ax in zip(indices, axes):
        image, true_label = dataset[idx]
        pred, confidence, probs = predict_single_image(model, image, device)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_display = image.cpu().numpy().transpose(1, 2, 0)
        img_display = (img_display * std + mean).clip(0, 1)

        ax.imshow(img_display)
        ax.axis('off')
        true_str = 'AD' if true_label == 1 else 'Normal'
        pred_str = 'AD' if pred == 1 else 'Normal'
        color = 'green' if pred == true_label else 'red'

        ax.set_title(f'True: {true_str} | Pred: {pred_str}\nConf: {confidence:.2f}', fontsize = 10, color = color, fontweight = 'bold')

    plt.suptitle('Sample Predictions', fontsize = 16, fontweight = 'bold', y = 0.98)
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

    bins = np.linspace(0, 1, 20)
    axes[0].hist(correct_conf, bins = bins, alpha = 0.7, label = 'Correct', color = 'green', edgecolor = 'black')
    axes[0].hist(wrong_conf, bins = bins, alpha = 0.7, label = 'Incorrect', color = 'red', edgecolor = 'black')
    axes[0].set_xlabel('Confidence', fontsize = 12)
    axes[0].set_ylabel('Count', fontsize = 12)
    axes[0].set_title('Confidence Distribution', fontsize = 14, fontweight = 'bold')
    axes[0].legend()
    axes[0].grid(True, alpha = 0.3)

    data = [correct_conf, wrong_conf]
    bp = axes[1].boxplot(data, labels = ['Correct', 'Incorrect'], patch_artist = True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    axes[1].set_ylabel('Confidence', fontsize = 12)
    axes[1].set_title('Confidence Comparison', fontsize = 14, fontweight = 'bold')
    axes[1].grid(True, alpha = 0.3, axis = 'y')
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

    fig, axes = plt.subplots(1, 2, figsize = (14, 5))
    labels = ['Correct', 'Incorrect']
    sizes = [correct, total - correct]
    colors = ['lightgreen', 'lightcoral']
    explode = (0.05, 0)

    axes[0].pie(sizes, explode = explode, labels = labels, colors = colors, autopct = '%1.1f%%', startangle = 90, textprops = {'fontsize': 12, 'fontweight': 'bold'})
    axes[0].set_title(f'Overall Accuracy: {correct / total:.1%}', fontsize = 14, fontweight = 'bold')
    error_types = ['True Pos', 'False Pos', 'False Neg', 'True Neg']
    counts = [true_pos, false_pos, false_neg, true_neg]
    colors = ['green', 'red', 'orange', 'green']

    axes[1].bar(error_types, counts, color = colors, alpha = 0.7)
    axes[1].set_ylabel('Count', fontsize = 12)
    axes[1].set_title('Error Type Analysis', fontsize = 14, fontweight = 'bold')

    for i, count in enumerate(counts):
        axes[1].text(i, count + 0.5, str(count), ha = 'center', fontsize = 12, fontweight = 'bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
        plt.close()
    else:
        plt.show()

# ------------------------------------------------------------------------------------------------------------------
def run_predictions():
    print("=" * 80)
    print("RUNNING PREDICTIONS ON TEST SET")
    print("=" * 80)
    
    model = load_trained_model(PredictConfig.pathOfModel, PredictConfig.DEVICE)

    test_dataset = ADNIDataset(
        data_root = PredictConfig.rootOfData,
        mode = 'test',
        transform = get_transforms('test', PredictConfig.IMG_SIZE),
        img_size = PredictConfig.IMG_SIZE
    )

    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size = 8, shuffle = False, num_workers = 8)
    results = predict_batch(model, test_loader, PredictConfig.DEVICE)

    correct = sum(1 for r in results if r['true_label'] == r['prediction'])
    total = len(results)
    accuracy = correct / total
    
    # Calculate additional metrics
    true_pos = sum(1 for r in results if r['true_label'] == 1 and r['prediction'] == 1)
    false_pos = sum(1 for r in results if r['true_label'] == 0 and r['prediction'] == 1)
    false_neg = sum(1 for r in results if r['true_label'] == 1 and r['prediction'] == 0)
    true_neg = sum(1 for r in results if r['true_label'] == 0 and r['prediction'] == 0)
    
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0

    # Generate plots
    print("\n📊 Generating visualization plots...")
    visualize_multiple_predictions(model, test_dataset, num_samples = 12, save_path = PredictConfig.outputDirectory / 'prediction_grid.png')
    plot_confidence_distribution(results, save_path = PredictConfig.outputDirectory / 'confidence_distribution.png')
    plot_error_analysis(results, save_path = PredictConfig.outputDirectory / 'error_analysis.png')

    # Find examples
    correct_normal = next((i, r) for i, r in enumerate(results) if r['true_label'] == 0 and r['prediction'] == 0)
    correct_ad = next((i, r) for i, r in enumerate(results) if r['true_label'] == 1 and r['prediction'] == 1)

    image, label = test_dataset[correct_normal[0]]
    pred, conf, probs = predict_single_image(model, image, PredictConfig.DEVICE)
    visualise_prediction(image, label, pred, probs, save_path = PredictConfig.outputDirectory / 'example_correct_normal.png')

    image, label = test_dataset[correct_ad[0]]
    pred, conf, probs = predict_single_image(model, image, PredictConfig.DEVICE)
    visualise_prediction(image, label, pred, probs, save_path = PredictConfig.outputDirectory / 'example_correct_ad.png')

    wrong_examples = [(i, r) for i, r in enumerate(results) if r['true_label'] != r['prediction']]
    if wrong_examples:
        image, label = test_dataset[wrong_examples[0][0]]
        pred, conf, probs = predict_single_image(model, image, PredictConfig.DEVICE)
        visualise_prediction(image, label, pred, probs, save_path = PredictConfig.outputDirectory / 'example_wrong_prediction.png')

    # Print results
    print("\n" + "=" * 80)
    print("🎯 TEST SET RESULTS")
    print("=" * 80)
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall * 100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision * 100:.2f}%)")
    print("=" * 80)
    
    if accuracy >= 0.8 and recall >= 0.8:
        print("✅ BOTH TARGETS ACHIEVED! (Accuracy & Recall > 80%)")
    elif accuracy >= 0.8:
        print("⚠️  Accuracy target met, but recall below 80%")
    elif recall >= 0.8:
        print("⚠️  Recall target met, but accuracy below 80%")
    else:
        print("❌ Below target (need both Accuracy & Recall > 80%)")
    
    print(f"\n✅ All visualizations saved to {PredictConfig.outputDirectory}")

# ------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    run_predictions()