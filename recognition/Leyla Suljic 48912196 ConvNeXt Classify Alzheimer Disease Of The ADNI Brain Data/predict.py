# ------------------------------------------------------------------------------------------------------------------
#  ConvNeXt Inference & Visualization Script
# ------------------------------------------------------------------------------------------------------------------
#  This script loads the trained model and performs inference on test samples.
#  Includes visualization of predictions, attention maps, and model interpretability.
# ------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import json
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Import our modules
from dataset import ADNIDataset, get_transforms
from modules import ConvNeXtAlzheimer


# ------------------------------------------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------------------------------------------
class PredictConfig:
    """Prediction configuration"""

    # Paths
    MODEL_PATH = './alzheimer_results/checkpoints/best_model_final.pth'
    DATA_ROOT = './ADNI/AD_NC'
    OUTPUT_DIR = Path('./alzheimer_results/predictions')

    # Settings
    BATCH_SIZE = 1  # Process one at a time for visualization
    IMG_SIZE = 224
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


# ------------------------------------------------------------------------------------------------------------------
# MODEL LOADER
# ------------------------------------------------------------------------------------------------------------------
def load_trained_model(model_path, device='cuda'):
    """Load the trained ConvNeXt model"""

    print(f"Loading model from {model_path}...")

    checkpoint = torch.load(model_path, map_location=device)

    # Recreate model architecture
    # Note: These should match the training config!
    model = ConvNeXtAlzheimer(
        model_size='tiny',  # Change if you used 'small' or 'base'
        num_classes=2,
        pretrained=False,  # We're loading trained weights
        dropout=0.3
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully!")
    print(f"  Trained for: {checkpoint['epoch']} epochs")
    print(f"  Val Accuracy: {checkpoint['val_acc']:.4f}")

    return model


# ------------------------------------------------------------------------------------------------------------------
# SINGLE IMAGE PREDICTION
# ------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def predict_single_image(model, image_tensor, device='cuda'):
    """
    Predict on a single image.

    Returns:
        prediction (int): 0=Normal, 1=AD
        confidence (float): Confidence score (0-1)
        probabilities (dict): {'Normal': prob, 'AD': prob}
    """
    model.eval()

    # Add batch dimension if needed
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    # Forward pass
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
# BATCH PREDICTION
# ------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def predict_batch(model, dataloader, device='cuda'):
    """
    Predict on an entire dataset.

    Returns:
        results (list): List of dicts with predictions
    """
    model.eval()
    results = []

    print("Running batch predictions...")

    for images, labels in tqdm(dataloader):
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
# VISUALIZATION FUNCTIONS
# ------------------------------------------------------------------------------------------------------------------
def visualize_prediction(image_tensor, true_label, pred_label, probabilities, save_path=None):
    """
    Visualize a single prediction with probabilities.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Denormalize image for display
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image_tensor.cpu().numpy().transpose(1, 2, 0)
    image = (image * std + mean).clip(0, 1)

    # Display image
    axes[0].imshow(image)
    axes[0].axis('off')

    # Add label
    true_label_str = 'AD' if true_label == 1 else 'Normal'
    pred_label_str = 'AD' if pred_label == 1 else 'Normal'
    color = 'green' if true_label == pred_label else 'red'

    axes[0].set_title(f'True: {true_label_str} | Pred: {pred_label_str}',
                      fontsize=14, fontweight='bold', color=color)

    # Plot probabilities
    classes = ['Normal', 'AD']
    probs = [probabilities['Normal'], probabilities['AD']]
    colors = ['skyblue', 'salmon']

    bars = axes[1].barh(classes, probs, color=colors, alpha=0.7)
    axes[1].set_xlim([0, 1])
    axes[1].set_xlabel('Probability', fontsize=12)
    axes[1].set_title('Class Probabilities', fontsize=14, fontweight='bold')

    # Add probability labels on bars
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        axes[1].text(prob + 0.02, bar.get_y() + bar.get_height() / 2,
                     f'{prob:.3f}', va='center', fontsize=11, fontweight='bold')

    # Add confidence
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


def visualize_multiple_predictions(model, dataset, num_samples=12, save_path=None):
    """
    Visualize predictions for multiple samples in a grid.
    """
    device = PredictConfig.DEVICE

    # Randomly sample indices
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    rows = 3
    cols = num_samples // rows
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()

    for idx, ax in zip(indices, axes):
        image, true_label = dataset[idx]

        # Predict
        pred, confidence, probs = predict_single_image(model, image, device)

        # Denormalize image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_display = image.cpu().numpy().transpose(1, 2, 0)
        img_display = (img_display * std + mean).clip(0, 1)

        # Display
        ax.imshow(img_display)
        ax.axis('off')

        # Title with prediction
        true_str = 'AD' if true_label == 1 else 'Normal'
        pred_str = 'AD' if pred == 1 else 'Normal'
        color = 'green' if pred == true_label else 'red'

        ax.set_title(f'True: {true_str} | Pred: {pred_str}\nConf: {confidence:.2f}',
                     fontsize=10, color=color, fontweight='bold')

    plt.suptitle('Sample Predictions', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_confidence_distribution(results, save_path=None):
    """
    Plot distribution of prediction confidences.
    """
    correct_confs = [r['confidence'] for r in results if r['true_label'] == r['prediction']]
    wrong_confs = [r['confidence'] for r in results if r['true_label'] != r['prediction']]

    plt.figure(figsize=(10, 6))

    plt.hist(correct_confs, bins=20, alpha=0.7, label=f'Correct (n={len(correct_confs)})', color='green')
    plt.hist(wrong_confs, bins=20, alpha=0.7, label=f'Wrong (n={len(wrong_confs)})', color='red')

    plt.xlabel('Confidence', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add statistics
    if correct_confs:
        mean_correct = np.mean(correct_confs)
        plt.axvline(mean_correct, color='green', linestyle='--', linewidth=2,
                    label=f'Mean Correct: {mean_correct:.3f}')

    if wrong_confs:
        mean_wrong = np.mean(wrong_confs)
        plt.axvline(mean_wrong, color='red', linestyle='--', linewidth=2,
                    label=f'Mean Wrong: {mean_wrong:.3f}')

    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_error_analysis(results, save_path=None):
    """
    Analyze which cases the model gets wrong.
    """
    # Separate by true label and prediction
    true_normal_pred_normal = sum(1 for r in results if r['true_label'] == 0 and r['prediction'] == 0)
    true_normal_pred_ad = sum(1 for r in results if r['true_label'] == 0 and r['prediction'] == 1)
    true_ad_pred_normal = sum(1 for r in results if r['true_label'] == 1 and r['prediction'] == 0)
    true_ad_pred_ad = sum(1 for r in results if r['true_label'] == 1 and r['prediction'] == 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Confusion matrix style
    cm_data = np.array([[true_normal_pred_normal, true_normal_pred_ad],
                        [true_ad_pred_normal, true_ad_pred_ad]])

    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Pred: Normal', 'Pred: AD'],
                yticklabels=['True: Normal', 'True: AD'])
    axes[0].set_title('Prediction Matrix', fontsize=14, fontweight='bold')

    # Error types
    error_types = ['True Negative\n(Correct)', 'False Positive\n(Type I Error)',
                   'False Negative\n(Type II Error)', 'True Positive\n(Correct)']
    counts = [true_normal_pred_normal, true_normal_pred_ad, true_ad_pred_normal, true_ad_pred_ad]
    colors = ['green', 'red', 'orange', 'green']

    axes[1].bar(error_types, counts, color=colors, alpha=0.7)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Error Type Analysis', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=0)

    # Add counts on bars
    for i, (count, color) in enumerate(zip(counts, colors)):
        axes[1].text(i, count + 0.5, str(count), ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ------------------------------------------------------------------------------------------------------------------
# MAIN PREDICTION PIPELINE
# ------------------------------------------------------------------------------------------------------------------
def run_predictions():
    """Main prediction and visualization pipeline"""

    print("=" * 80)
    print("CONVNEXT ALZHEIMER'S CLASSIFIER - INFERENCE")
    print("=" * 80)

    # Load model
    model = load_trained_model(PredictConfig.MODEL_PATH, PredictConfig.DEVICE)

    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = ADNIDataset(
        data_root=PredictConfig.DATA_ROOT,
        mode='test',
        transform=get_transforms('test', PredictConfig.IMG_SIZE),
        img_size=PredictConfig.IMG_SIZE
    )

    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

    # Run batch predictions
    print("\n" + "=" * 80)
    print("RUNNING PREDICTIONS ON TEST SET")
    print("=" * 80)

    results = predict_batch(model, test_loader, PredictConfig.DEVICE)

    # Calculate metrics
    correct = sum(1 for r in results if r['true_label'] == r['prediction'])
    total = len(results)
    accuracy = correct / total

    print(f"\nTest Set Results:")
    print(f"  Total Samples: {total}")
    print(f"  Correct: {correct}")
    print(f"  Wrong: {total - correct}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Save results
    results_path = PredictConfig.OUTPUT_DIR / 'test_predictions.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n✓ Results saved to {results_path}")

    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    # 1. Multiple predictions grid
    print("Creating prediction grid...")
    visualize_multiple_predictions(
        model, test_dataset, num_samples=12,
        save_path=PredictConfig.OUTPUT_DIR / 'prediction_grid.png'
    )
    print("✓ Saved: prediction_grid.png")

    # 2. Confidence distribution
    print("Plotting confidence distribution...")
    plot_confidence_distribution(
        results,
        save_path=PredictConfig.OUTPUT_DIR / 'confidence_distribution.png'
    )
    print("✓ Saved: confidence_distribution.png")

    # 3. Error analysis
    print("Creating error analysis...")
    plot_error_analysis(
        results,
        save_path=PredictConfig.OUTPUT_DIR / 'error_analysis.png'
    )
    print("✓ Saved: error_analysis.png")

    # 4. Individual predictions for each class
    print("Creating individual prediction examples...")

    # Find examples of each type
    correct_normal = next((i, r) for i, r in enumerate(results)
                          if r['true_label'] == 0 and r['prediction'] == 0)
    correct_ad = next((i, r) for i, r in enumerate(results)
                      if r['true_label'] == 1 and r['prediction'] == 1)

    # Visualize correct predictions
    image, label = test_dataset[correct_normal[0]]
    pred, conf, probs = predict_single_image(model, image, PredictConfig.DEVICE)
    visualize_prediction(image, label, pred, probs,
                         save_path=PredictConfig.OUTPUT_DIR / 'example_correct_normal.png')
    print("✓ Saved: example_correct_normal.png")

    image, label = test_dataset[correct_ad[0]]
    pred, conf, probs = predict_single_image(model, image, PredictConfig.DEVICE)
    visualize_prediction(image, label, pred, probs,
                         save_path=PredictConfig.OUTPUT_DIR / 'example_correct_ad.png')
    print("✓ Saved: example_correct_ad.png")

    # Find wrong predictions if they exist
    wrong_examples = [(i, r) for i, r in enumerate(results) if r['true_label'] != r['prediction']]
    if wrong_examples:
        image, label = test_dataset[wrong_examples[0][0]]
        pred, conf, probs = predict_single_image(model, image, PredictConfig.DEVICE)
        visualize_prediction(image, label, pred, probs,
                             save_path=PredictConfig.OUTPUT_DIR / 'example_wrong_prediction.png')
        print("✓ Saved: example_wrong_prediction.png")

    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE! 🎉")
    print("=" * 80)
    print(f"\nAll results saved to: {PredictConfig.OUTPUT_DIR}")
    print(f"\nFinal Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    if accuracy >= 0.8:
        print("✓ TARGET ACCURACY ACHIEVED! (>0.8)")
    else:
        print("⚠ Below target accuracy (0.8)")


# ------------------------------------------------------------------------------------------------------------------
# INTERACTIVE PREDICTION (FOR DEMO)
# ------------------------------------------------------------------------------------------------------------------
def interactive_demo():
    """
    Interactive demo for live predictions.
    Useful for demonstrating to instructors!
    """
    print("=" * 80)
    print("INTERACTIVE ALZHEIMER'S CLASSIFICATION DEMO")
    print("=" * 80)

    # Load model
    model = load_trained_model(PredictConfig.MODEL_PATH, PredictConfig.DEVICE)

    # Load test dataset
    test_dataset = ADNIDataset(
        data_root=PredictConfig.DATA_ROOT,
        mode='test',
        transform=get_transforms('test', PredictConfig.IMG_SIZE),
        img_size=PredictConfig.IMG_SIZE
    )

    print(f"\nLoaded {len(test_dataset)} test samples")
    print("\nInstructions:")
    print("  - Enter a sample index (0-{}) to classify".format(len(test_dataset) - 1))
    print("  - Enter 'random' for a random sample")
    print("  - Enter 'quit' to exit")

    while True:
        user_input = input("\nEnter sample index: ").strip().lower()

        if user_input == 'quit':
            break

        if user_input == 'random':
            idx = np.random.randint(0, len(test_dataset))
        else:
            try:
                idx = int(user_input)
                if idx < 0 or idx >= len(test_dataset):
                    print(f"Invalid index! Must be between 0 and {len(test_dataset) - 1}")
                    continue
            except ValueError:
                print("Invalid input!")
                continue

        # Load and predict
        image, true_label = test_dataset[idx]
        pred, confidence, probs = predict_single_image(model, image, PredictConfig.DEVICE)

        # Display results
        print("\n" + "-" * 60)
        print(f"Sample #{idx}")
        print(f"  True Label: {'AD' if true_label == 1 else 'Normal'}")
        print(f"  Prediction: {'AD' if pred == 1 else 'Normal'}")
        print(f"  Confidence: {confidence:.1%}")
        print(f"  Probabilities:")
        print(f"    - Normal: {probs['Normal']:.3f}")
        print(f"    - AD:     {probs['AD']:.3f}")
        print(f"  Result: {'✓ CORRECT' if pred == true_label else '✗ WRONG'}")
        print("-" * 60)

        # Visualize
        visualize_prediction(image, true_label, pred, probs)

    print("\nDemo ended. Thanks for trying!")


# ------------------------------------------------------------------------------------------------------------------
# RUN PREDICTIONS
# ------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        interactive_demo()
    else:
        run_predictions()