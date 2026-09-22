"""LP-SRNet: longitudinal super-resolution for line-structured-light depth images.

The network keeps the SMFANet feature-modulation backbone and changes three
parts, matching the TDSR article:

* one intermediate block is a geometry-aware feature modulation block (GAFMB);
* that block uses edge attention and a partial deformable convolution
  feed-forward network;
* reconstruction is progressive 2x pixel shuffle along the longitudinal axis.

Image layout is (N, C, H, W). H is the longitudinal (travel) axis and W is the
transverse axis. Scales 4 and 8 therefore change H only.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

from basicsr.archs.SMFANet_arch import FMB, PCFN, SMFA, DMlp
from basicsr.utils.registry import ARCH_REGISTRY


class LongitudinalPixelShuffle(nn.Module):
    """Move channels onto the longitudinal axis and leave width unchanged."""

    def __init__(self, scale):
        super().__init__()
        if scale < 2:
            raise ValueError(f'Longitudinal pixel shuffle scale must be >= 2, got {scale}.')
        self.scale = int(scale)

    def forward(self, x):
        b, c, h, w = x.shape
        scale = self.scale
        if c % scale != 0:
            raise RuntimeError(f'Channel count {c} is not divisible by longitudinal scale {scale}.')
        c_out = c // scale
        x = x.view(b, c_out, scale, h, w)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        return x.view(b, c_out, h * scale, w)


class EdgeAttention(nn.Module):
    """Multi-scale depthwise edges, pooled self-attention, and channel gating.

    Channels are split in half. The first half uses a 3x3 depthwise branch and
    the second half uses a 5x5 depthwise branch. Their concatenation is max-pooled
    to a fixed grid, attended, and turned into a sigmoid channel gate.
    """

    def __init__(self, dim, pool_size=8):
        super().__init__()
        if dim % 4 != 0:
            raise ValueError(f'Edge attention expects a channel count divisible by 4, got {dim}.')
        half = dim // 2
        self.dw3 = nn.Conv2d(half, half, 3, 1, 1, groups=half)
        self.pw3 = nn.Conv2d(half, half, 1, 1, 0)
        self.dw5 = nn.Conv2d(half, half, 5, 1, 2, groups=half)
        self.pw5 = nn.Conv2d(half, half, 1, 1, 0)
        self.act = nn.GELU()
        self.pool_size = int(pool_size)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.fc1 = nn.Linear(dim, dim // 4)
        self.fc2 = nn.Linear(dim // 4, dim)
        self.scale = dim ** -0.5

    def forward(self, x):
        b, c, h, w = x.shape
        x3, x5 = x.chunk(2, dim=1)
        fe = torch.cat((self.act(self.pw3(self.dw3(x3))), self.act(self.pw5(self.dw5(x5)))), dim=1)
        pooled = F.adaptive_max_pool2d(fe, (min(self.pool_size, h), min(self.pool_size, w)))
        tokens = pooled.flatten(2).transpose(1, 2)
        q = self.q(tokens)
        k = self.k(tokens)
        v = self.v(tokens)
        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        gap = (attn @ v).mean(dim=1)
        gate = torch.sigmoid(self.fc2(self.act(self.fc1(gap))))
        return x * gate[:, :, None, None]


class EGSMFA(nn.Module):
    """Edge-guided SMFA: local detail branch plus an edge-attention branch."""

    def __init__(self, dim=36, pool_size=8):
        super().__init__()
        self.linear_0 = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.lde = DMlp(dim, 2)
        self.ea = EdgeAttention(dim, pool_size)

    def forward(self, f):
        y, x = self.linear_0(f).chunk(2, dim=1)
        return self.linear_2(self.ea(x) + self.lde(y))


class PartialDeformConv2d(nn.Module):
    """3x3 deformable convolution with an 18-channel offset field and no mask."""

    def __init__(self, channels):
        super().__init__()
        self.offset = nn.Conv2d(channels, 18, 3, 1, 1)
        self.weight = nn.Parameter(torch.empty(channels, channels, 3, 3))
        self.bias = nn.Parameter(torch.zeros(channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, x):
        return deform_conv2d(x, self.offset(x), self.weight, self.bias, padding=1)


class PDCFN(nn.Module):
    """Partial deformable convolution feed-forward network.

    A 1x1 projection and GELU are split into a C/4 deformable branch and a 3C/4
    bypass. The deformable branch is activated again, concatenated, and fused
    by a 1x1 convolution.
    """

    def __init__(self, dim):
        super().__init__()
        if dim % 4 != 0:
            raise ValueError(f'PDCFN expects a channel count divisible by 4, got {dim}.')
        self.p_dim = dim // 4
        self.conv_in = nn.Conv2d(dim, dim, 1, 1, 0)
        self.pcdc = PartialDeformConv2d(self.p_dim)
        self.conv_out = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.conv_in(x))
        x1, x2 = torch.split(x, [self.p_dim, x.shape[1] - self.p_dim], dim=1)
        x1 = self.act(self.pcdc(x1))
        return self.conv_out(torch.cat((x1, x2), dim=1))


class GAFMB(nn.Module):
    """Geometry-aware feature modulation block used once inside LP-SRNet."""

    def __init__(self, dim, ffn_scale=2.0, use_ea=True, use_pdcfn=True, pool_size=8):
        super().__init__()
        self.aggregator = EGSMFA(dim, pool_size) if use_ea else SMFA(dim)
        self.ffn = PDCFN(dim) if use_pdcfn else PCFN(dim, ffn_scale)

    def forward(self, x):
        x = self.aggregator(F.normalize(x)) + x
        x = self.ffn(F.normalize(x)) + x
        return x


class ProgressiveLongitudinalUpsample(nn.Module):
    """Two 2x stages for 4x restoration, or three 2x stages for 8x.

    Each stage applies a depthwise 3x3 refinement and a pointwise projection
    that emits two longitudinal subpixel groups, then rearranges those groups
    along height. Intermediate stages keep the feature width unchanged.
    """

    def __init__(self, dim, scale, out_channels):
        super().__init__()
        n_stages = int(round(math.log2(scale)))
        if 2 ** n_stages != scale:
            raise ValueError(f'Progressive longitudinal shuffle expects a power-of-two scale, got {scale}.')
        stages = []
        for stage_idx in range(n_stages):
            expanded = out_channels * 2 if stage_idx == n_stages - 1 else dim * 2
            stages.append(nn.Sequential(
                nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
                nn.Conv2d(dim, expanded, 1, 1, 0),
                LongitudinalPixelShuffle(2),
            ))
        self.stages = nn.ModuleList(stages)

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x


@ARCH_REGISTRY.register()
class LPSRNet(nn.Module):
    """Line-structured-light pavement super-resolution network.

    Args:
        dim (int): Feature width. Default: 36.
        n_blocks (int): Number of modulation blocks. Default: 8, of which one
            intermediate block is the GAFMB when edge attention or PDCFN is on.
        ffn_scale (float): Expansion ratio of the original partial convolution FFN.
        upscaling_factor (int): Longitudinal scale, 4 or 8.
        in_channels (int): Depth input channels. Default: 1.
        out_channels (int): Reconstructed depth channels. Default: 1.
        gafmb_index (int): Index of the intermediate GAFMB. Default: 3.
        use_ea (bool): Replace that block's nonlocal branch with edge attention.
        use_pdcfn (bool): Replace that block's feed-forward network with PDCFN.
        use_pps (bool): Use progressive longitudinal pixel shuffle. If False, one
            convolution performs the full 4x or 8x longitudinal rearrangement.
        pool_size (int): Fixed spatial size of the edge-attention grid.
    """

    def __init__(self,
                 dim=36,
                 n_blocks=8,
                 ffn_scale=2.0,
                 upscaling_factor=4,
                 in_channels=1,
                 out_channels=1,
                 gafmb_index=3,
                 use_ea=True,
                 use_pdcfn=True,
                 use_pps=True,
                 pool_size=8):
        super().__init__()
        if upscaling_factor not in (4, 8):
            raise ValueError(f'LP-SRNet supports longitudinal scales 4 and 8, got {upscaling_factor}.')
        if not 0 <= gafmb_index < n_blocks:
            raise ValueError(f'gafmb_index must lie in [0, {n_blocks}), got {gafmb_index}.')

        self.scale = int(upscaling_factor)
        self.to_feat = nn.Conv2d(in_channels, dim, 3, 1, 1)
        blocks = []
        for block_idx in range(n_blocks):
            if block_idx == gafmb_index and (use_ea or use_pdcfn):
                blocks.append(GAFMB(dim, ffn_scale, use_ea, use_pdcfn, pool_size))
            else:
                blocks.append(FMB(dim, ffn_scale))
        self.feats = nn.Sequential(*blocks)
        if use_pps:
            self.to_img = ProgressiveLongitudinalUpsample(dim, self.scale, out_channels)
        else:
            self.to_img = nn.Sequential(
                nn.Conv2d(dim, out_channels * self.scale, 3, 1, 1),
                LongitudinalPixelShuffle(self.scale),
            )

    def forward(self, x):
        x = self.to_feat(x)
        x = self.feats(x) + x
        return self.to_img(x)
