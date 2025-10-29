import torch
import torch.nn as nn
import timm

class MultiBranchHead(nn.Module):
    """
    Implements the FULL multi-branch head from the paper with 3 branches:
    1. Global Average Pooling (captures holistic features)
    2. Global Max Pooling (captures salient features)
    3. Attention-weighted Pooling (learns to focus on important regions)
    
    This is the KEY improvement - we now have all 3 branches!
    """
    def __init__(self, in_features, dropout=0.3):
        super(MultiBranchHead, self).__init__()
        
        # Branch 1: Global Average Pooling
        self.pool_avg = nn.AdaptiveAvgPool2d(1)
        
        # Branch 2: Global Max Pooling
        self.pool_max = nn.AdaptiveMaxPool2d(1)
        
        # Branch 3: Attention-weighted Pooling (NEW!)
        # Learn an attention mask to focus on important regions
        self.attention_conv = nn.Sequential(
            nn.Conv2d(in_features, in_features // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features // 8, 1, kernel_size=1),
            nn.Sigmoid()  # Output attention weights in [0, 1]
        )
        
        # Now we have 3 branches concatenated: in_features * 3
        self.in_features_tripled = in_features * 3
        
        self.flatten = nn.Flatten()
        
        # Feature Selection Layer (from paper)
        # This learns to weight the importance of combined features
        self.feature_selection = nn.Sequential(
            nn.Linear(self.in_features_tripled, self.in_features_tripled),
            nn.Sigmoid()  # Gating mechanism
        )
        
        # Classifier head (matching paper's architecture more closely)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.in_features_tripled, 512),
            nn.BatchNorm1d(512),  # Added batch norm for stability
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, 256),  # Added intermediate layer
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 3),
            nn.Linear(256, 2)  # 2 output classes (NC, AD)
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x is the feature map from the backbone (e.g., [B, 1024, 7, 7])
        
        # Branch 1: Global Average Pooling
        x_avg = self.pool_avg(x)  # [B, 1024, 1, 1]
        
        # Branch 2: Global Max Pooling
        x_max = self.pool_max(x)  # [B, 1024, 1, 1]
        
        # Branch 3: Attention-weighted Pooling (KEY ADDITION!)
        attention_weights = self.attention_conv(x)  # [B, 1, H, W]
        x_weighted = x * attention_weights  # Apply attention
        x_att = self.pool_avg(x_weighted)  # [B, 1024, 1, 1]
        
        # Combine all 3 branches
        x_combined = torch.cat([x_avg, x_max, x_att], dim=1)  # [B, 3072, 1, 1]
        
        # Flatten
        x_flat = self.flatten(x_combined)  # [B, 3072]
        
        # Feature Selection Layer (learns importance weights)
        x_gated = self.feature_selection(x_flat) * x_flat  # Element-wise gating
        
        # Final classification
        return self.classifier(x_gated)


class ConvNeXtAlzheimer(nn.Module):
    """
    Wrapper class that matches the interface expected by predict.py
    """
    def __init__(self, model_size='base', num_classes=2, pretrained=True, dropout=0.3):
        super(ConvNeXtAlzheimer, self).__init__()
        
        if model_size == 'base':
            model_name = 'convnext_base'
        elif model_size == 'small':
            model_name = 'convnext_small'
        else:
            model_name = 'convnext_tiny'
        
        # Load backbone
        backbone = timm.create_model(model_name, pretrained=pretrained, in_chans=3, num_classes=0)
        
        # Extract components
        self.stem = backbone.stem
        self.stages = backbone.stages
        self.num_features = backbone.num_features
        
        # Attach our 3-branch head
        self.head = MultiBranchHead(self.num_features, dropout)
        
    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.head(x)
        return x
    
    def freeze_backbone(self, freeze=True, unfreeze_blocks=2):
        """Helper method to freeze/unfreeze backbone"""
        print("\n" + "-"*50)
        if freeze:
            print("❄️ Freezing backbone weights (Phase 1)...")
            # Freeze everything
            for param in self.stem.parameters():
                param.requires_grad = False
            for stage in self.stages:
                for param in stage.parameters():
                    param.requires_grad = False
            # Unfreeze head
            for param in self.head.parameters():
                param.requires_grad = True
            print("   Trainable: model.head only")
        else:
            print(f"🔥 Unfreezing top {unfreeze_blocks} blocks (Phase 2)...")
            # First freeze everything
            for param in self.stem.parameters():
                param.requires_grad = False
            for stage in self.stages:
                for param in stage.parameters():
                    param.requires_grad = False
            
            # Unfreeze last N blocks
            num_stages = len(self.stages)
            num_to_freeze = num_stages - unfreeze_blocks
            
            for i in range(num_to_freeze, num_stages):
                for param in self.stages[i].parameters():
                    param.requires_grad = True
            
            # Always keep head trainable
            for param in self.head.parameters():
                param.requires_grad = True
            
            print(f"   Frozen: stem, stages 0..{num_to_freeze-1}")
            print(f"   Trainable: stages {num_to_freeze}..{num_stages-1}, head")
        print("-" * 50)


def create_alzheimer_model(model_size='base', pretrained=True, dropout=0.3, device='cuda'):
    """
    Creates the ConvNeXt model with 3-branch head
    """
    model = ConvNeXtAlzheimer(
        model_size=model_size,
        num_classes=2,
        pretrained=pretrained,
        dropout=dropout
    )
    return model.to(device)