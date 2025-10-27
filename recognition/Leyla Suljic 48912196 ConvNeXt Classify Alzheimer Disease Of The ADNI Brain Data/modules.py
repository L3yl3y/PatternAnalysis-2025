# ------------------------------------------------------------------------------------------------------------------
#  ConvNeXt Model for Alzheimer's Disease Classification
# ------------------------------------------------------------------------------------------------------------------
#  This module implements ConvNeXt (2022) - a modern CNN that BEATS vision transformers while being simpler!
#
#  Why ConvNeXt for medical imaging?
#  - Excellent transfer learning from ImageNet (already knows general visual features)
#  - Pure CNN architecture = better for small datasets (doesn't need as much data as transformers)
#  - Hierarchical feature extraction (captures both fine details AND global patterns)
#  - State-of-the-art accuracy with reasonable compute
#
#  ConvNeXt Innovations:
#  - Depthwise convolutions (more efficient)
#  - Larger kernels (7x7 like transformers' large receptive fields)
#  - LayerNorm instead of BatchNorm (more stable)
#  - GELU activation (smoother than ReLU)
# ------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights, ConvNeXt_Base_Weights


# ------------------------------------------------------------------------------------------------------------------
# CONVNEXT MODEL BUILDER
# ------------------------------------------------------------------------------------------------------------------
class ConvNeXtAlzheimer(nn.Module):
    """
    ConvNeXt for binary Alzheimer's classification (Normal vs AD).

    Uses transfer learning: We start with ConvNeXt pretrained on ImageNet (1.2M images),
    then fine-tune it on our ADNI brain scans. This is CRITICAL because:
    - ADNI only has ~400 samples (tiny!)
    - Training from scratch would overfit like crazy
    - Pretrained features (edges, textures) transfer well to medical images
    """

    def __init__(self, model_size='tiny', num_classes=2, pretrained=True, dropout=0.3):
        """
        Args:
            model_size: 'tiny', 'small', or 'base' (bigger = more accurate but slower)
            num_classes: 2 for binary classification (Normal vs AD)
            pretrained: Use ImageNet weights (ALWAYS True for medical imaging!)
            dropout: Dropout rate for regularization (prevents overfitting)
        """
        super().__init__()

        self.model_size = model_size
        self.num_classes = num_classes

        # Load pretrained ConvNeXt backbone
        print(f"Loading ConvNeXt-{model_size.upper()} with pretrained weights...")

        if model_size == 'tiny':
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.convnext_tiny(weights=weights)
            self.feature_dim = 768  # ConvNeXt-Tiny output dimension

        elif model_size == 'small':
            weights = ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.convnext_small(weights=weights)
            self.feature_dim = 768  # ConvNeXt-Small output dimension

        elif model_size == 'base':
            weights = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.convnext_base(weights=weights)
            self.feature_dim = 1024  # ConvNeXt-Base output dimension

        else:
            raise ValueError(f"Invalid model_size: {model_size}. Choose 'tiny', 'small', or 'base'")

        # Replace the classification head
        # Original head is for 1000 ImageNet classes, we need 2 classes
        self.backbone.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),  # Flatten first! [B, C, 1, 1] -> [B, C]
            nn.LayerNorm(self.feature_dim, eps=1e-6),  # Normalize features
            nn.Dropout(dropout),  # Regularization (randomly drop neurons)
            nn.Linear(self.feature_dim, 512),  # Intermediate layer
            nn.GELU(),  # GELU activation (smoother than ReLU, used in ConvNeXt)
            nn.Dropout(dropout / 2),  # Less dropout in deeper layer
            nn.Linear(512, num_classes)  # Final classification
        )

        # Initialize new layers with good values
        self._init_classification_head()

        print(f"✓ ConvNeXt-{model_size.upper()} loaded successfully!")
        print(f"  Feature dim: {self.feature_dim}")
        print(f"  Dropout: {dropout}")
        print(f"  Output classes: {num_classes}")

    def _init_classification_head(self):
        """
        Initialize the new classification head with Xavier initialization.
        This gives the model a good starting point for learning.
        """
        for m in self.backbone.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through the network.

        Input: (batch_size, 3, 224, 224) - RGB-like brain slices
        Output: (batch_size, 2) - logits for [Normal, AD]
        """
        return self.backbone(x)

    def freeze_backbone(self, freeze=True):
        """
        Freeze/unfreeze the backbone layers.

        Training strategy:
        1. First train with backbone frozen (only train classifier head) - FAST
        2. Then unfreeze and fine-tune entire model - better accuracy

        This is called "gradual unfreezing" and prevents catastrophic forgetting
        of the pretrained ImageNet features.
        """
        for param in self.backbone.features.parameters():
            param.requires_grad = not freeze

        status = "frozen" if freeze else "unfrozen"
        print(f"Backbone parameters {status}")

    def get_num_params(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_feature_maps(self, x):
        """
        Extract intermediate feature maps for visualization.
        Useful for understanding what the model "sees" in brain scans.
        """
        features = []

        # Hook function to capture features
        def hook_fn(module, input, output):
            features.append(output)

        # Register hooks on different stages
        hooks = []
        for i, stage in enumerate(self.backbone.features):
            hook = stage.register_forward_hook(hook_fn)
            hooks.append(hook)

        # Forward pass
        with torch.no_grad():
            _ = self.forward(x)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return features


# ------------------------------------------------------------------------------------------------------------------
# MODEL FACTORY FUNCTION
# ------------------------------------------------------------------------------------------------------------------
def create_alzheimer_model(model_size='tiny', pretrained=True, dropout=0.3, device='cuda'):
    """
    Factory function to create and configure ConvNeXt model.

    Recommended configurations:
    - For fast experimentation: tiny, dropout=0.3
    - For best accuracy: small or base, dropout=0.4
    - If overfitting: increase dropout to 0.5
    """
    model = ConvNeXtAlzheimer(
        model_size=model_size,
        num_classes=2,
        pretrained=pretrained,
        dropout=dropout
    )

    model = model.to(device)

    print(f"\nModel Statistics:")
    print(f"  Total parameters: {model.get_num_params():,}")
    print(f"  Model size: ~{model.get_num_params() * 4 / 1024 / 1024:.1f} MB")

    return model


# ------------------------------------------------------------------------------------------------------------------
# ENSEMBLE MODEL (OPTIONAL - FOR EVEN BETTER ACCURACY)
# ------------------------------------------------------------------------------------------------------------------
class ConvNeXtEnsemble(nn.Module):
    """
    Ensemble of multiple ConvNeXt models for improved accuracy.

    Ensemble = multiple models vote on the final prediction.
    Like having 3 doctors instead of 1 - more reliable!

    Typically improves accuracy by 2-3% but 3x slower.
    """

    def __init__(self, model_sizes=['tiny', 'tiny', 'tiny'], dropout=0.3):
        super().__init__()

        self.models = nn.ModuleList([
            ConvNeXtAlzheimer(size, num_classes=2, pretrained=True, dropout=dropout)
            for size in model_sizes
        ])

        print(f"Created ensemble of {len(self.models)} models: {model_sizes}")

    def forward(self, x):
        """Average predictions from all models"""
        outputs = [model(x) for model in self.models]
        return torch.stack(outputs).mean(dim=0)

    def freeze_backbone(self, freeze=True):
        """Freeze all model backbones"""
        for model in self.models:
            model.freeze_backbone(freeze)


# ------------------------------------------------------------------------------------------------------------------
# ATTENTION MODULE (OPTIONAL ENHANCEMENT)
# ------------------------------------------------------------------------------------------------------------------
class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism to focus on important brain regions.

    In Alzheimer's, certain regions matter more:
    - Hippocampus (memory)
    - Temporal lobe
    - Ventricles

    This module learns to weight these regions higher.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.sigmoid(self.conv(x))
        return x * attention


class ConvNeXtWithAttention(ConvNeXtAlzheimer):
    """ConvNeXt + Spatial Attention for improved focus on diagnostic regions"""

    def __init__(self, model_size='tiny', num_classes=2, pretrained=True, dropout=0.3):
        super().__init__(model_size, num_classes, pretrained, dropout)

        # Add attention after last conv stage
        self.attention = SpatialAttention(self.feature_dim)
        print("✓ Added spatial attention mechanism")

    def forward(self, x):
        # Extract features
        features = self.backbone.features(x)

        # Apply attention
        features = self.attention(features)

        # Classification head
        pooled = self.backbone.avgpool(features)
        output = self.backbone.classifier(pooled)

        return output


# ------------------------------------------------------------------------------------------------------------------
# TESTING CODE
# ------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    print("Testing ConvNeXt Model...")

    # Test model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test tiny model
    model_tiny = create_alzheimer_model('tiny', pretrained=True, device=device)

    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    output = model_tiny(dummy_input)

    print(f"\nTest forward pass:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")  # Should be (4, 2)
    print(f"  Output: {output}")

    # Test predictions
    probs = F.softmax(output, dim=1)
    preds = torch.argmax(probs, dim=1)
    print(f"  Probabilities: {probs}")
    print(f"  Predictions: {preds} (0=Normal, 1=AD)")

    # Test freezing
    model_tiny.freeze_backbone(freeze=True)
    model_tiny.freeze_backbone(freeze=False)

    print("\n✓ Model tests passed!")

    # Test attention model
    print("\nTesting attention model...")
    model_attn = ConvNeXtWithAttention('tiny', pretrained=True, dropout=0.3)
    model_attn = model_attn.to(device)
    output_attn = model_attn(dummy_input)
    print(f"  Attention output shape: {output_attn.shape}")

    print("\n✓ All model tests successful!")