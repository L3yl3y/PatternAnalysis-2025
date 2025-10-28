import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights, ConvNeXt_Base_Weights

# ==================================================================================================================
# This model we were told to select is like the brain if you will so its pretty smart and already has a lot of ...
# pretrained images that allow it to be able to understand these brain images (like edges, shapes, textures etc).
# So one thing that stands out about this is that it uses a two-phase training approach referred to as ...
# Freezing and Unfreezing the backbone.
# Pretty much this file creates a neural network that looks at brain MRIs, extracts features (patterns, textures)...
# It will then decide whether that brain image is Normal or Alzheimers?
# ==================================================================================================================
class ConvNeXtAlzheimer(nn.Module):
    # num_classes = 2 since there are only 2 clasifications (Normal vs AD).
    # Cheeky little bit of dropout for learning - will turn off neurons during training to try prevent overfitting.
    def __init__(self, model_size='tiny', num_classes = 2, pretrained = True, dropout = 0.3):
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
        
        self.backbone.classifier = nn.Sequential( # I used this in demo and it worked well so I love sequential.
            nn.Flatten(start_dim = 1), # Flatten spatial dimensions; must be 1D input not 2D maps.
            nn.LayerNorm(self.feature_dim, eps = 1e-6), # Normalize features to mean = 0, std = 1; this allows for better stability and convergence. 
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.GELU(), # Adds non-linearity (similar to ReLU like in demo2).
            nn.Dropout(dropout / 2), # Use less dropout in the deeper layers.
            nn.Linear(512, num_classes)
        )
        
        self._init_classification_head()
    
    def _init_classification_head(self):
        # This is new content but it is suggested to use Xavier initialisation since this will find all the linear layers
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
    
    def freeze_backbone(self, freeze = True):
        """
        Freeze or unfreeze the pre-trained backbone layers.
        
        What "freezing" means:
        - Frozen layers: Weights DON'T change during training
        - Unfrozen layers: Weights DO change during training
        
        Training strategy (Two-Phase Training):
        
        Phase 1 (Backbone FROZEN):
        ├─ ConvNeXt backbone: FROZEN ❄️ (weights locked)
        └─ Classification head: UNFROZEN 🔥 (weights update)
        
        Why? Train only the new head first (fast, prevents breaking pretrained features)
        Duration: ~10 epochs
        
        Phase 2 (Everything UNFROZEN):
        ├─ ConvNeXt backbone: UNFROZEN 🔥 (fine-tune features)
        └─ Classification head: UNFROZEN 🔥 (continue learning)
        
        Why? Fine-tune the entire network for best accuracy
        Duration: ~30-40 epochs
        
        Benefits of two-phase training:
        ✅ Phase 1 is fast (only ~10% of params train)
        ✅ Prevents "catastrophic forgetting" of ImageNet features
        ✅ Better final accuracy than training everything at once
        ✅ More stable training
        
        Args:
            freeze (bool): True = freeze backbone, False = unfreeze backbone
        
        Example usage in train.py:
            # Phase 1: Train only classification head
            model.freeze_backbone(freeze=True)
            train_for_10_epochs()
            
            # Phase 2: Fine-tune entire model
            model.freeze_backbone(freeze=False)
            train_for_40_epochs()
        """
        # Loop through all parameters in the backbone feature extractor
        for param in self.backbone.features.parameters():
            # parameters() returns all weight tensors
            # self.backbone.features = the ConvNeXt stages (not the classifier)
            
            # requires_grad = "should PyTorch compute gradients for this parameter?"
            # If requires_grad=False → frozen (no learning)
            # If requires_grad=True → unfrozen (will learn)
            param.requires_grad = not freeze
            # If freeze=True:  requires_grad=False (frozen)
            # If freeze=False: requires_grad=True (unfrozen)
        
        # Print status so we know what happened
        status = "frozen" if freeze else "unfrozen"
        print(f"Backbone parameters {status}")
        
        # Note: We only freeze/unfreeze the BACKBONE (features)
        # The classification head is ALWAYS unfrozen (always learning)
    
    def get_num_params(self):
        """
        Count the total number of trainable parameters.
        
        What are parameters?
        - Weights and biases in all layers
        - These are the values that the model learns during training
        
        Why count them?
        - Know model size (bigger = more memory needed)
        - Understand computational cost
        - Compare different model configurations
        
        Returns:
            int: Total number of trainable parameters
        
        Example output:
            ConvNeXt-Tiny:  ~28,000,000 parameters (~28M)
            ConvNeXt-Small: ~50,000,000 parameters (~50M)
            ConvNeXt-Base:  ~89,000,000 parameters (~89M)
        """
        # Sum up all parameter counts
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Breakdown:
        # self.parameters() → all parameter tensors in model
        # if p.requires_grad → only count trainable params (skip frozen ones)
        # p.numel() → number of elements in tensor
        # sum(...) → add them all up
        #
        # Example for one Linear layer with input=768, output=512:
        # Weight: [512, 768] = 393,216 parameters
        # Bias:   [512]      = 512 parameters
        # Total:              = 393,728 parameters

def create_alzheimer_model(model_size = 'base', pretrained = True, dropout = 0.3, device = 'cuda'):
    # This creates model and prints some pretty statistics.
    # Setting model_size = 'base' will make my model better.
    """
    Factory function to create and configure a ConvNeXt model.
    
    This is a convenience function that:
    1. Creates the model
    2. Moves it to GPU/CPU
    3. Prints useful statistics
    4. Returns ready-to-use model
    
    Args:
        model_size (str): 'tiny', 'small', or 'base'
            Recommendations:
            - 'tiny': Fast experiments, good for laptops (~28M params)
            - 'small': Balanced, good for most cases (~50M params)
            - 'base': Best accuracy, needs good GPU (~89M params)
        
        pretrained (bool): Use ImageNet pre-trained weights?
            Should ALWAYS be True for medical imaging!
            Starting from scratch would need 10,000+ images
        
        dropout (float): Dropout rate (0.0 to 1.0)
            - 0.3: Good default, balanced regularization
            - 0.4-0.5: If overfitting (training acc >> val acc)
            - 0.2: If underfitting (both accs low)
        
        device (str): 'cuda' for GPU or 'cpu' for CPU
            GPU is 10-100× faster!
            If you don't have GPU, training takes hours instead of minutes
    
    Returns:
        model: ConvNeXtAlzheimer model ready for training
    
    Example usage:
        # Create model
        model = create_alzheimer_model('tiny', device='cuda')
        
        # Use in training
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            ...
    """
    model = ConvNeXtAlzheimer(
        model_size = model_size,
        num_classes = 2,
        pretrained = pretrained, 
        dropout = dropout
    )

    model = model.to(device)
    return model