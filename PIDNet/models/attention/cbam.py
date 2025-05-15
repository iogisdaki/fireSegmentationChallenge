import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# CHANNEL ATTENTION MODULE
# (From Section 3.1 and Fig. 3 of the CBAM paper)
# ----------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super().__init__()
        # Adaptive average pooling (Eq. 1, first pooling type)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Adaptive max pooling (Eq. 1, second pooling type)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP: implemented as two 1x1 conv layers (Fig. 3)
        self.shared = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),  # reduction
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)   # expansion
        )

        # Sigmoid gating (Eq. 1, final step)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply average and max pooling
        avg = self.shared(self.avg_pool(x))  # F_avg in Eq.1
        max = self.shared(self.max_pool(x))  # F_max in Eq.1
        # Element-wise addition and sigmoid gating (Eq.1)
        return self.sigmoid(avg + max)


# ----------------------------
# SPATIAL ATTENTION MODULE
# (From Section 3.2 and Fig. 4 of the CBAM paper)
# ----------------------------
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # Single 7x7 convolution (Fig. 4)
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        # Sigmoid activation (Eq. 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply average pooling along channel axis (Eq. 2)
        avg = torch.mean(x, dim=1, keepdim=True)  # F_avg^s
        # Apply max pooling along channel axis (Eq. 2)
        max, _ = torch.max(x, dim=1, keepdim=True)  # F_max^s
        # Concatenate along channel axis (Fig. 4)
        concat = torch.cat((avg, max), dim=1)  # shape: [B, 2, H, W]
        # Pass through 7x7 conv and sigmoid gate (Eq. 2)
        return self.sigmoid(self.conv(concat))


# ----------------------------
# CBAM Module (Channel + Spatial)
# (From Fig. 2 of the CBAM paper — Full attention block)
# ----------------------------
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super().__init__()
        # Channel attention submodule (first applied)
        self.ca = ChannelAttention(in_planes, ratio)
        # Spatial attention submodule (second applied)
        self.sa = SpatialAttention()

    def forward(self, x):
        # Multiply input by channel attention map (Eq. 1)
        out = x * self.ca(x)
        # Multiply result by spatial attention map (Eq. 2)
        out = out * self.sa(out)
        # Final enhanced feature map
        return out
