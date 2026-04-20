"""
CNN-Transformer Based Recognition Model
Architecture: CNN Feature Extractor → Flatten+Linear → Transformer Encoder → FC Classifier
Reference: Table 1 - Implementation Parameters of the Recognition Module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Algorithm 1: CNN-Transformer Recognition Model
# =============================================================================
# Input:  Image tensor of shape (batch_size, C, H, W)
# Output: Class logits of shape (batch_size, num_classes)
#
# Stage 1 — CNN Feature Extraction:
#   Conv1: in=1,   out=32,  kernel=3, padding=1 → ReLU → Pool(2, stride=2)
#   Conv2: in=32,  out=64,  kernel=3, padding=1 → ReLU → Pool(2, stride=2)
#   Conv3: in=64,  out=128, kernel=3, padding=1 → ReLU → Pool(2, stride=2)
#
# Stage 2 — CNN→Transformer Transition:
#   Flatten spatial dims  → shape (batch, 128*H'*W')
#   Linear projection     → shape (batch, 128)          [128×8×8 → 128]
#
# Stage 3 — Transformer Encoder:
#   Input  128-dim sequence → Transformer Encoder Layer → 128-dim output
#
# Stage 4 — Classification Head:
#   FC1: 128 → 512  + ReLU + Dropout
#   FC2: 512 → 125  (raw logits)
# =============================================================================


class ConvBlock(nn.Module):
    """Single convolutional block: Conv → ReLU → MaxPool."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, padding: int = 1,
                 pool_size: int = 2, pool_stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.conv(x)))


class CNNFeatureExtractor(nn.Module):
    """
    Three-stage CNN backbone.

    Progressively extracts visual features from coarse (edges/textures)
    to abstract representations while reducing spatial dimensions via pooling.

    Input  shape: (B, 1,   H,    W   )
    Output shape: (B, 128, H/8,  W/8 )   [3 pooling ops, each ÷2]
    """

    def __init__(self):
        super().__init__()
        # Table 1 — Conv layers
        self.block1 = ConvBlock(in_channels=1,   out_channels=32,
                                kernel_size=3, padding=1,
                                pool_size=2, pool_stride=2)   # /2
        self.block2 = ConvBlock(in_channels=32,  out_channels=64,
                                kernel_size=3, padding=1,
                                pool_size=2, pool_stride=2)   # /2
        self.block3 = ConvBlock(in_channels=64,  out_channels=128,
                                kernel_size=3, padding=1,
                                pool_size=2, pool_stride=2)   # /2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)   # (B,  32, H/2,  W/2 )
        x = self.block2(x)   # (B,  64, H/4,  W/4 )
        x = self.block3(x)   # (B, 128, H/8,  W/8 )
        return x


class CNNToTransformerBridge(nn.Module):
    """
    Transition block: flattens CNN spatial output and projects to
    the Transformer's expected embedding dimension.

    For a 64×64 input image the CNN outputs (B, 128, 8, 8).
    Flattened: 128 × 8 × 8 = 8192  →  Linear  →  128   (Table 1)
    """

    def __init__(self, cnn_out_channels: int = 128,
                 spatial_h: int = 8, spatial_w: int = 8,
                 transformer_dim: int = 128):
        super().__init__()
        flattened_dim = cnn_out_channels * spatial_h * spatial_w  # 8192
        self.flatten = nn.Flatten()
        self.linear  = nn.Linear(flattened_dim, transformer_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)    # (B, 128*8*8)
        x = self.linear(x)     # (B, 128)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer Encoder Layer.

    Uses multi-head self-attention so every position in the flattened
    feature vector can attend to all others, capturing global context.

    Input/Output shape: (B, transformer_dim)  →  unsqueeze seq_len=1
                         for nn.TransformerEncoderLayer compatibility.
    """

    def __init__(self, d_model: int = 128, nhead: int = 8,
                 dim_feedforward: int = 512, dropout: float = 0.1,
                 num_layers: int = 1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True      # (B, seq, dim)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 128) → add sequence dimension → (B, 1, 128)
        x = x.unsqueeze(1)
        x = self.transformer_encoder(x)   # (B, 1, 128)
        x = x.squeeze(1)                  # (B, 128)
        return x


class ClassificationHead(nn.Module):
    """
    Two-layer FC classification head.

    FC1: 128 → 512  + ReLU + Dropout
    FC2: 512 → 125  (logits — apply Softmax externally if probabilities needed)
    """

    def __init__(self, in_features: int = 128,
                 hidden_features: int = 512,
                 num_classes: int = 125,
                 dropout_prob: float = 0.5):
        super().__init__()
        self.fc1     = nn.Linear(in_features, hidden_features)
        self.relu    = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2     = nn.Linear(hidden_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))   # (B, 512)
        x = self.dropout(x)
        x = self.fc2(x)              # (B, 125)  — raw logits
        return x


class CNNTransformerRecognitionModel(nn.Module):
    """
    Full SNN-Transformer Recognition Model.

    Pipeline
    --------
    Input Image (B, 1, H, W)
        │
        ▼
    CNN Feature Extractor      [3 ConvBlocks: 1→32→64→128 channels]
        │  (B, 128, H/8, W/8)
        ▼
    CNN→Transformer Bridge     [Flatten + Linear: 128×8×8 → 128]
        │  (B, 128)
        ▼
    Transformer Encoder Layer  [Self-attention: 128 → 128]
        │  (B, 128)
        ▼
    Classification Head        [FC1: 128→512 + ReLU + Dropout]
        │                      [FC2: 512→125]
        ▼
    Logits (B, num_classes)

    Parameters
    ----------
    num_classes       : int   Number of output classes (default 125, per Table 1)
    input_h, input_w  : int   Expected spatial size after 3 poolings.
                              For 64×64 input → 8×8 after three ÷2 pools.
    nhead             : int   Number of self-attention heads (must divide d_model=128)
    dropout_prob      : float Dropout rate in classification head
    """

    def __init__(self,
                 num_classes: int = 125,
                 input_h: int = 8,
                 input_w: int = 8,
                 nhead: int = 8,
                 dropout_prob: float = 0.5):
        super().__init__()

        # Stage 1: CNN backbone
        self.cnn = CNNFeatureExtractor()

        # Stage 2: CNN → Transformer transition
        self.bridge = CNNToTransformerBridge(
            cnn_out_channels=128,
            spatial_h=input_h,
            spatial_w=input_w,
            transformer_dim=128
        )

        # Stage 3: Transformer encoder
        self.transformer = TransformerEncoderBlock(
            d_model=128,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1,
            num_layers=1
        )

        # Stage 4: Classification head
        self.classifier = ClassificationHead(
            in_features=128,
            hidden_features=512,
            num_classes=num_classes,
            dropout_prob=dropout_prob
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor  Shape (B, 1, H, W)  — grayscale input images

        Returns
        -------
        logits : torch.Tensor  Shape (B, num_classes)
        """
        # Stage 1 — CNN feature extraction
        x = self.cnn(x)             # (B, 128, H/8, W/8)

        # Stage 2 — Flatten + linear projection
        x = self.bridge(x)          # (B, 128)

        # Stage 3 — Transformer encoder (global context)
        x = self.transformer(x)     # (B, 128)

        # Stage 4 — FC classification head
        logits = self.classifier(x) # (B, 125)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities via Softmax (for inference)."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Training Utilities
# =============================================================================

def build_model(num_classes: int = 125,
                image_size: int = 64,
                nhead: int = 8,
                dropout_prob: float = 0.5) -> CNNTransformerRecognitionModel:
    """
    Convenience factory.

    Parameters
    ----------
    num_classes  : Target class count (Table 1: FC2 output = 125)
    image_size   : Square input size. Must be divisible by 8 (3 poolings of ÷2).
                   Common choices: 64 (→ 8×8), 32 (→ 4×4).
    nhead        : Attention heads; must divide 128.
    dropout_prob : Dropout rate in classification head.
    """
    assert image_size % 8 == 0, "image_size must be divisible by 8 (3× MaxPool2d with stride 2)"
    spatial = image_size // 8
    model = CNNTransformerRecognitionModel(
        num_classes=num_classes,
        input_h=spatial,
        input_w=spatial,
        nhead=nhead,
        dropout_prob=dropout_prob
    )
    return model


def get_default_optimizer(model: nn.Module,
                           lr: float = 1e-3,
                           weight_decay: float = 1e-4):
    """Adam optimizer — standard choice for Transformer-based models."""
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def get_default_scheduler(optimizer, num_epochs: int = 50):
    """Cosine annealing LR scheduler."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


# =============================================================================
# Quick Smoke Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  CNN-Transformer Recognition Model — Smoke Test")
    print("=" * 60)

    # Build model (default: 64×64 grayscale input, 125 classes)
    model = build_model(num_classes=125, image_size=64)
    model.eval()

    # Dummy batch: (batch=4, channels=1, height=64, width=64)
    dummy_input = torch.randn(4, 1, 64, 64)

    with torch.no_grad():
        logits = model(dummy_input)
        probs  = model.predict_proba(dummy_input)

    print(f"\nInput  shape : {dummy_input.shape}")
    print(f"Logits shape : {logits.shape}   (expected: [4, 125])")
    print(f"Probs  shape : {probs.shape}")
    print(f"Probs  sum   : {probs[0].sum().item():.6f}  (expected: 1.0)")
    print(f"\nTrainable parameters: {model.count_parameters():,}")

    # Per-stage parameter counts
    stages = {
        "CNN Extractor      ": model.cnn,
        "CNN→Transformer    ": model.bridge,
        "Transformer Encoder": model.transformer,
        "Classification Head": model.classifier,
    }
    print("\nPer-stage parameter counts:")
    for name, module in stages.items():
        n = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  {name}: {n:>10,}")

    print("\n✓ Forward pass successful.\n")

    # Print model summary
    print(model)