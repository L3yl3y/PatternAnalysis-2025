"""
GFNet Architecture for Alzheimer's Disease Classification
Adapted from successful implementations achieving 80%+ accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math


class Mlp(nn.Module):
    """Multi-Layer Perceptron with GELU activation"""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalFilter(nn.Module):
    """Global Filter Layer - operates in frequency domain"""
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(
            torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02
        )
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)
        x = x.to(torch.float32)

        # Apply FFT
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        
        # Multiply by learned complex weights
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        
        # Apply inverse FFT
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
        x = x.reshape(B, N, C)

        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class Block(nn.Module):
    """GFNet Block with Global Filter and MLP"""
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, h=14, w=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = GlobalFilter(dim, h=h, w=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class GFNet(nn.Module):
    """
    Global Filter Network for Alzheimer's Disease Classification
    Optimized for medical imaging with grayscale input
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=1, num_classes=2,
                 embed_dim=384, depth=12, mlp_ratio=4., drop_rate=0., 
                 drop_path_rate=0., norm_layer=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, 
            in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Calculate h and w for Global Filter
        h = img_size // patch_size
        w = h // 2 + 1

        # Transformer blocks with Global Filters
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, mlp_ratio=mlp_ratio,
                drop=drop_rate, drop_path=dpr[i], 
                norm_layer=norm_layer, h=h, w=w
            )
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        # Classification head with enhanced features
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x).mean(1)  # Global average pooling
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def create_gfnet_model(img_size=224, patch_size=16, in_chans=1, num_classes=2,
                       embed_dim=384, depth=12, drop_rate=0.1, drop_path_rate=0.1):
    """
    Factory function to create GFNet model
    
    Args:
        img_size: Input image size (default: 224)
        patch_size: Patch size for embedding (default: 16)
        in_chans: Number of input channels (1 for grayscale)
        num_classes: Number of output classes (2 for AD vs NC)
        embed_dim: Embedding dimension (384 for balanced performance)
        depth: Number of transformer blocks (12 for deep features)
        drop_rate: Dropout rate
        drop_path_rate: Stochastic depth rate
    """
    print("\n" + "="*70)
    print("CREATING GFNET MODEL FOR ALZHEIMER'S DETECTION")
    print("="*70)
    print(f"   Configuration:")
    print(f"   • Image size: {img_size}x{img_size}")
    print(f"   • Patch size: {patch_size}x{patch_size}")
    print(f"   • Input channels: {in_chans} (grayscale)")
    print(f"   • Embedding dim: {embed_dim}")
    print(f"   • Depth: {depth} blocks")
    print(f"   • Dropout: {drop_rate}")
    print(f"   • Drop path: {drop_path_rate}")
    
    model = GFNet(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        mlp_ratio=4.,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n   Model Statistics:")
    print(f"   • Total parameters: {total_params:,}")
    print(f"   • Trainable parameters: {trainable_params:,}")
    print("="*70 + "\n")
    
    return model