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
from torch.utils.data import DataLoader

"""
This is the config class which is meant to group all our settings into one place. So without needing to change a later
setting we can do so all in this one file (without needing to hunt through the code you see)."""
# There is a lot going on, so I added these segmenters ----- to isolate functions, so I don't get confused.
class PredictConfig:
    pathOfModel = './alzheimer_results_v3/checkpoints/best_model.pth' # Updated path for my v3 model/
    rootOfData = './data/ADNI/AD_NC'
    outputDirectory = Path('./alzheimer_results_v3/predictions')

    BATCH_SIZE = 1
    IMG_SIZE = 224 # Same image size model trains with.
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    outputDirectory.mkdir(exist_ok = True, parents = True) # Create output directory if it doesn't already exist.

# ------------------------------------------------------------------------------------------------------------------
"""Loads trained PyTorch model - re-builds model as it was during training (same size, same dropout etc) and then
 it will like load it in with the same saved weights and stuff from the file."""
def load_trained_model(model_path, device='cuda'):
    checkpoint = torch.load(model_path, map_location = device)

    model = ConvNeXtAlzheimer( # Skeleton of the og model (same architecture as one we trained v3 fyi).
        model_size = 'small',  # Updated from 'tiny'
        num_classes = 2,
        pretrained = False,
        dropout = 0.35
    )

    model.load_state_dict(checkpoint['model_state_dict']) # Put saved weights into skeleton model.
    model = model.to(device)
    model.eval() # Eval mode to turn off training layers and used learned stats for batch normalisation.
    # Lmao make sure not to forget this since this is what actually makes the predictions nor random and not incorrect.

    print(f"✔ Model loaded successfully!")
    print(f"  Model: ConvNeXt-small")
    print(f"  Trained for: {checkpoint['epoch']} epochs")
    
    # Hehe pretty printing for checkpoints so it helps me write README.
    if 'test_acc' in checkpoint:
        print(f"  Best Test Accuracy: {checkpoint['test_acc']:.4f}")

    if 'test_recall' in checkpoint:
        print(f"  Best Test Recall: {checkpoint['test_recall']:.4f}")

    if 'threshold' in checkpoint:
        print(f"  Decision Threshold: {checkpoint['threshold']:.3f}")
        return model, checkpoint['threshold']

    return model, 0.5 # The 0.5 confused me too but it's a safety case (it is for in case the threshold wasn't saved).

# ------------------------------------------------------------------------------------------------------------------
""" FYI the torch.no_grad() is an optimisation that tells PyTorch not to calculate gradients for this function. 
This is, from my understanding, because gradients are only needed for training (backpropagation) so if you turn them off 
it allows for the code to run much faster with less memory. Also this function here is what we use to make a prediction
on a single image tensor."""
@torch.no_grad()
def predict_single_image(model, image_tensor, device = 'cuda', threshold=0.5):
    model.eval()

    # Because models are designed to work on batches of images, a single image tensor only has 3 dimensions.
    # The unsqueeze(0)  will add a fake batch dimension of 1 to make 3 dimensions [C, H, W] into [1, C, H, W].
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)
    outputs = model(image_tensor) # Forward pass --> feed image to model (outputs are raw scores).
    probs = F.softmax(outputs, dim = 1) # Softmax to turn those raw output scores into probabilities.
    ad_prob = probs[0, 1].item() # .item() will pull the single number from PyTorch tensor that is probability of AD.
    prediction = 1 if ad_prob >= threshold else 0
    confidence = probs[0, prediction].item()

    probabilities = { # Dict to hold both probabilities (normal brains and sick brains).
        'Normal': probs[0, 0].item(),
        'AD': probs[0, 1].item()
    }

    return prediction, confidence, probabilities

# ------------------------------------------------------------------------------------------------------------------
"""Runs prediction on 'entire' dataset using DataLoader --> the main workhorse if you will. So looping through all
the data, gets predictions in batches and gets all the results collected into one big list for analysis."""
@torch.no_grad() # Same as above - we know its not to calculate gradients.
def predict_batch(model, dataloader, device = 'cuda', threshold = 0.5):
    model.eval() # Note this is just a good practice thing for safety (set model to eval in case I forget).
    results = []

    # Similar logic wise to predict single image but now for the batch.
    # FYI tqdm is a pretty progress bar teehee - this loop is what gets the images (i.e., 8 at a time).
    for images, labels in tqdm(dataloader, desc = 'Making predictions'):
        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim = 1)
        ad_probs = probs[:, 1]
        preds = (ad_probs >= threshold).long() # This is a fast, parallel awy to apply threshold to entire batch.
        # Looks like (ad_probs >= threshold) --> [True, False, True, ...] --> and .long() causes --> [1, 0, 1, ...].

        for i in range(len(images)):
            results.append({ # Store all results in a dict like this.
                'true_label': labels[i].item(),
                'prediction': preds[i].item(),
                'confidence': probs[i, preds[i]].item(),
                'prob_normal': probs[i, 0].item(),
                'prob_ad': probs[i, 1].item()
            })

    return results

# ------------------------------------------------------------------------------------------------------------------
""" 2-panel plot to see single prediction. I promise you there is nothing exciting down here just slop to print for
 visual comparison to gauge what is going right/wrong with my model."""
def visualise_prediction(image_tensor, true_label, pred_label, probabilities, save_path = None):
    fig, axes = plt.subplots(1, 2, figsize = (12, 5))
    mean = np.array([0.485, 0.456, 0.406]) # Our get_transforms function will normalise the images.
    std = np.array([0.229, 0.224, 0.225]) # Un-normalise to see the pics/images normally.
    image = image_tensor.cpu().numpy().transpose(1, 2, 0)
    image = (image * std + mean).clip(0, 1) # Un-normalisation math.
    axes[0].imshow(image)
    axes[0].axis('off') # Hide ugly x/y axes bahah.
    
    true_label_str = 'AD' if true_label == 1 else 'Normal'
    pred_label_str = 'AD' if pred_label == 1 else 'Normal'
    color = 'green' if true_label == pred_label else 'red'
    axes[0].set_title(f'True: {true_label_str} | Pred: {pred_label_str}',
                      fontsize = 14, fontweight = 'bold', color = color)

    # Get the data for the bar chart here.
    classes = ['Normal', 'AD']
    probs = [probabilities['Normal'], probabilities['AD']]
    colors = ['skyblue', 'salmon']
    bars = axes[1].barh(classes, probs, color = colors, alpha = 0.7)
    axes[1].set_xlim([0, 1])
    axes[1].set_xlabel('Probability', fontsize = 12)
    axes[1].set_title('Class Probabilities', fontsize = 14, fontweight = 'bold')

    # This loop here adds the exact probability value as text on the plot.
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        axes[1].text(prob + 0.02, bar.get_y() + bar.get_height() / 2,
                    f'{prob:.3f}', va = 'center', fontsize = 11, fontweight = 'bold')
        # Note, for the sake of formatting I had made sure everything was below the line for code characters.

    confidence = max(probs)
    axes[1].text(0.5, -0.15, f'Confidence: {confidence:.1%}', ha = 'center', fontsize = 12, fontweight = 'bold',
                transform = axes[1].transAxes)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
        plt.close() # This is important to close the plot to free up memory.
    else:
        plt.show()

# ------------------------------------------------------------------------------------------------------------------
""" Does the correct vs incorrect predictions distribution - this is the graph with the green and red bars."""
def plot_confidence_distribution(results, save_path = None):
    correct_conf = [r['confidence'] for r in results if r['true_label'] == r['prediction']] # List comprehension.
    wrong_conf = [r['confidence'] for r in results if r['true_label'] != r['prediction']]

    # HISTOGRAM:
    fig, axes = plt.subplots(1, 2, figsize = (14, 5))
    bins = np.linspace(0, 1, 20)
    axes[0].hist(correct_conf, bins = bins, alpha = 0.7, 
                 label = f'Correct (n={len(correct_conf)})', color = 'green', edgecolor = 'black') # Correct.
    axes[0].hist(wrong_conf, bins = bins, alpha = 0.7, 
                 label = f'Incorrect (n={len(wrong_conf)})', color = 'red', edgecolor = 'black') # Incorrect.
    axes[0].set_xlabel('Confidence', fontsize = 12)
    axes[0].set_ylabel('Count', fontsize = 12)
    axes[0].set_title('Confidence Distribution', fontsize = 14, fontweight = 'bold')
    axes[0].legend()
    axes[0].grid(True, alpha = 0.3)

    # BOXPLOT:
    data = [correct_conf, wrong_conf]
    bp = axes[1].boxplot(data, labels = ['Correct', 'Incorrect'], patch_artist = True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    axes[1].set_ylabel('Confidence', fontsize = 12)
    axes[1].set_title('Confidence Comparison', fontsize = 14, fontweight = 'bold')
    axes[1].grid(True, alpha = 0.3, axis = 'y')
    
    # Add mean value as text on boxplot.
    for i, d in enumerate(data):
        if len(d) > 0: # Safety here to avoid errors if the list so happens to be empty.
            mean_val = np.mean(d)
            axes[1].text(i+1, mean_val, f'μ = {mean_val:.3f}',
                        ha = 'center', va = 'bottom', fontweight = 'bold')
    
    plt.tight_layout()

    # Same logic as earlier.
    if save_path:
        plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
        plt.close()
    else:
        plt.show()

# ------------------------------------------------------------------------------------------------------------------
""" This is my favourite graph to really gauge what is going wrong, this pertains to the confusion matrix which has 4
slices that allows me to determine the correct, the false positives and the false negatives); good classification.
This is the bar charts + the confusion matrix namely - the most important imo."""
def plot_error_analysis(results, save_path = None):
    correct = sum(1 for r in results if r['true_label'] == r['prediction'])
    total = len(results)

    # These here are where you begin to calculate these metrics I was referring to in the javadoc:
    true_pos = sum(1 for r in results if r['true_label'] == 1 and r['prediction'] == 1)
    false_pos = sum(1 for r in results if r['true_label'] == 0 and r['prediction'] == 1)
    false_neg = sum(1 for r in results if r['true_label'] == 1 and r['prediction'] == 0)
    true_neg = sum(1 for r in results if r['true_label'] == 0 and r['prediction'] == 0)
    fig, axes = plt.subplots(1, 3, figsize = (18, 5))
    
    # Pie chart (Overall Accuracy):
    labels = ['Correct', 'Incorrect']
    sizes = [correct, total - correct]
    colors = ['lightgreen', 'lightcoral']
    explode = (0.05, 0)

    axes[0].pie(sizes, explode = explode, labels = labels, colors = colors, 
                autopct = '%1.1f%%', startangle = 90, 
                textprops = {'fontsize': 12, 'fontweight': 'bold'})
    axes[0].set_title(f'Overall Accuracy: {correct / total:.1%}', 
                     fontsize = 14, fontweight = 'bold')
    
    # Here is where we get tech with it and start doing the breakdown for false positives, false negatives, etc.
    # Very useful imo to truly see the raw numbers so you can see the respective outcome for how many right/wrong.
    error_types = ['True Pos', 'False Pos', 'False Neg', 'True Neg']
    counts = [true_pos, false_pos, false_neg, true_neg]
    colors = ['green', 'red', 'orange', 'blue']
    axes[1].bar(error_types, counts, color = colors, alpha = 0.7)
    axes[1].set_ylabel('Count', fontsize = 12)
    axes[1].set_title('Classification Breakdown', fontsize = 14, fontweight = 'bold')

    # This is what adds the cont number atop each bar.
    for i, count in enumerate(counts):
        axes[1].text(i, count + 0.5, str(count), ha = 'center', 
                    fontsize = 12, fontweight = 'bold')

    # Confusion Matrix - this is the 2x2 with the blues.
    # I LOVE THE CONFUSION MATRIX TOO - SIMILAR TO BAR GRAPH ABOVE WITH SAME STATISTICS:
    # NOTE THE ORDER:
    #   [TN, FP]
    #   [FN, TP]
    cm = [[true_neg, false_pos], [false_neg, true_pos]] # This here is the 2x2 Matrix:
    sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', ax = axes[2],
                xticklabels = ['Normal', 'AD'], yticklabels = ['Normal', 'AD'])
    axes[2].set_xlabel('Predicted', fontsize = 12)
    axes[2].set_ylabel('Actual', fontsize = 12)
    axes[2].set_title('Confusion Matrix', fontsize = 14, fontweight = 'bold')
    plt.tight_layout()

    # Same saving logic:
    if save_path:
        plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
        plt.close()
    else:
        plt.show()

# ------------------------------------------------------------------------------------------------------------------
""" More user metrics that ties everything together in the correct order and is the entry point function that ties 
everything together): the order being to load the model, load the test data, run the predictions on all the test data,
calculate the performance metrics, generate and save all the analysis plots and print the final user friendly formatted 
report to the console."""
def run_predictions():
    print("=" * 80)
    print("RUNNING PREDICTIONS WITH V3 OPTIMIZED MODEL :3") # Smiley face here.
    print("=" * 80)

    # (1) call the function to get the model and the threshold:
    model, optimal_threshold = load_trained_model(PredictConfig.pathOfModel, PredictConfig.DEVICE)
    print(f"  Using threshold: {optimal_threshold:.3f}")

    # (2) Create the Dataset Object for our test data --> we apply simple transforms (no augmentation).
    test_dataset = ADNIDataset(
        data_root = PredictConfig.rootOfData,
        mode = 'test',
        transform = get_transforms('test', PredictConfig.IMG_SIZE),
        img_size = PredictConfig.IMG_SIZE
    )

    # (3) Run predictions:
    # I know I could take out the emojis but I really like them :3
    test_loader = DataLoader(test_dataset, batch_size = 8, shuffle = False, num_workers = 8)
    print("\n📊 Evaluating on test set...")
    results = predict_batch(model, test_loader, PredictConfig.DEVICE, threshold = optimal_threshold)
    correct = sum(1 for r in results if r['true_label'] == r['prediction'])
    total = len(results)
    accuracy = correct / total

    true_pos = sum(1 for r in results if r['true_label'] == 1 and r['prediction'] == 1)
    false_pos = sum(1 for r in results if r['true_label'] == 0 and r['prediction'] == 1)
    false_neg = sum(1 for r in results if r['true_label'] == 1 and r['prediction'] == 0)
    true_neg = sum(1 for r in results if r['true_label'] == 0 and r['prediction'] == 0)

    # Fyi, the recall is, of all the people who have AD, what percentage did we actually correctly find?
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    specificity = true_neg / (true_neg + false_pos) if (true_neg + false_pos) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    # Note f1 is the mean of precision and recall (score that balances both)

    # (4) Generate visualisations:
    print("\n📊 Generating visualisation plots...")
    plot_confidence_distribution(results, save_path = PredictConfig.outputDirectory / 'confidence_distribution.png')
    plot_error_analysis(results, save_path = PredictConfig.outputDirectory / 'error_analysis.png')

    # ------------------------------------------------------------------------------------------------------------------
    # (5) Print detailed results:
    print("\n" + "=" * 80)
    print("🎯 TEST SET RESULTS (V3 OPTIMIZED MODEL)")
    print("=" * 80)
    print(f"  Total samples: {total}")
    print(f"  Correct predictions: {correct}")
    print(f"  Incorrect predictions: {total - correct}")
    print("\n📊 Performance Metrics:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"  Recall: {recall:.4f} ({recall * 100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision * 100:.2f}%)")
    print(f"  Specificity: {specificity:.4f} ({specificity * 100:.2f}%)")
    print(f"  F1 Score: {f1:.4f} ({f1 * 100:.2f}%)")
    print("\n📊 Confusion Matrix:")
    print(f"  True Positives:  {true_pos} (Correctly identified AD)")
    print(f"  True Negatives:  {true_neg} (Correctly identified Normal)")
    print(f"  False Positives: {false_pos} (Normal misclassified as AD)")
    print(f"  False Negatives: {false_neg} (AD misclassified as Normal)")
    print("=" * 80)

# ------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    run_predictions()