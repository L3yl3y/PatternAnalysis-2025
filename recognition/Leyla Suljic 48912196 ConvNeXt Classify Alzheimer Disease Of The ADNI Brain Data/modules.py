import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights, ConvNeXt_Base_Weights

""" ==================================================================================================================
# This model we were told to select is like the brain if you will so it's pretty smart and already has a lot of ...
# pretrained images that allow it to be able to understand these brain images (like edges, shapes, textures etc).
# So one thing that stands out about this is that it uses a two-phase training approach referred to as ...
# Freezing and Unfreezing the backbone.
# Pretty much this file creates a neural network that looks at brain MRIs, extracts features (patterns, textures)...
# It will then decide whether that brain image is Normal or Alzheimers?
# =================================================================================================================="""
class ConvNeXtAlzheimer(nn.Module):
    """num_classes = 2 since there are only 2 classifications (Normal vs AD).
    Cheeky little bit of dropout for learning - will turn off neurons during training to prevent overfitting."""
    def __init__(self, model_size = 'tiny', num_classes = 2, pretrained = True, dropout = 0.3):
        super().__init__()
        self.model_size = model_size
        self.num_classes = num_classes
        
        if model_size == 'tiny': # Load pre-trained ConvNeXt backbone based on size.
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.convnext_tiny(weights = weights)
            self.feature_dim = 768
            
        elif model_size == 'small':
            weights = ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.convnext_small(weights = weights)
            self.feature_dim = 768
            
        elif model_size == 'base':
            weights = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.convnext_base(weights=weights)
            self.feature_dim = 1024
        
        self.backbone.classifier = nn.Sequential( # I used this in demo, and it worked well so I love sequential.
            nn.Flatten(start_dim = 1), # Flatten spatial dimensions; must be 1D input not 2D maps.
            nn.LayerNorm(self.feature_dim, eps = 1e-6), # Normalise features to mean = 0, std = 1.
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.GELU(), # Adds non-linearity (similar to ReLU like in demo2).
            nn.Dropout(dropout / 2), # Use less dropout in the deeper layers.
            nn.Linear(512, num_classes)
        )
        
        self._init_classification_head()

    
    def _init_classification_head(self):
        # This is new content, but it is suggested to use Xavier initialisation since this will find all the linear layers
        # In our classification head and will prevent vanishing/exploding gradients - scaling weight based on the layer size.
        # Purportedly with Xavier initialisation the weights will be 'just right' and enable more stable training.
        for m in self.backbone.classifier.modules():
            if isinstance(m, nn.Linear): # Iterates through all modules and if it is linear then initialise it.
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Forward pass that will be how the data flows through the network; inputs 16 brain images and then outputs 16 predictions.
        # The prediction scores will have both normal and AD scores.
        # So the flow is brain image input --> backbone ConvNeXt stage --> classification head --> output logic --> convert to probability.
        # E.x., output = [2.3, -1.7] which has E.x., [92%, 8%] for Normal : AD.
        return self.backbone(x) # Pytorch will do all this scary stuff btw.

    # There is this freeze and unfreeze pre-training thing (they call it a two phase training).
    # So freeze = weights don't change (no learning).
    # So unfreeze = weights change (learning).
    def freeze_backbone(self, freeze = True):
        for param in self.backbone.features.parameters():
            param.requires_grad = not freeze

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def create_alzheimer_model(model_size = 'base', pretrained = True, dropout = 0.3, device = 'cuda'):
    model = ConvNeXtAlzheimer(
        model_size = model_size,
        num_classes = 2,
        pretrained = pretrained, 
        dropout = dropout
    )

    model = model.to(device)
    return model