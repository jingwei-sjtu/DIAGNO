# coding=utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_harmonics.sht import RealSHT, InverseRealSHT
    from timm.models.layers import DropPath 
except ImportError:
    DropPath = nn.Identity
    print("Warning: torch_harmonics or timm not found.")

from .layers.UNet_Blocks import DoubleConv2D, OutConv2D


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.wr = nn.Parameter(torch.Tensor(out_features, in_features))
        self.wi = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.br = nn.Parameter(torch.zeros(out_features))
            self.bi = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('br', None); self.register_parameter('bi', None)
        nn.init.xavier_uniform_(self.wr); nn.init.xavier_uniform_(self.wi)

    def forward(self, x):
        res_r = F.linear(x.real, self.wr) - F.linear(x.imag, self.wi)
        res_i = F.linear(x.real, self.wi) + F.linear(x.imag, self.wr)
        if self.br is not None:
            res_r, res_i = res_r + self.br, res_i + self.bi
        return torch.complex(res_r, res_i)


class ComplexLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.complex64))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        x_c = x - mean
        var = (x_c.real**2 + x_c.imag**2).mean(dim=-1, keepdim=True)
        return (x_c * torch.rsqrt(var + self.eps)) * self.weight + self.bias


class ChannelLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ComplexRealAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

    def forward(self, q, k, v):
        B, N, C = q.shape
        L_kv = k.shape[1]
        
        q = q.view(B, N, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, L_kv, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, L_kv, self.heads, self.head_dim).permute(0, 2, 1, 3)
        
        attn_score = ((q.real @ k.real.transpose(-2, -1)) + (q.imag @ k.imag.transpose(-2, -1))) * self.scale
        attn = attn_score.softmax(dim=-1)
                
        res = torch.complex(attn @ v.real, attn @ v.imag)
        return res.transpose(1, 2).reshape(B, N, C)


class SpectralTransolverEngine(nn.Module):
    def __init__(self, lmax, mmax, embed_dim):
        super().__init__()

        self.lmax = lmax
        self.mmax = mmax

        self.complex_project = ComplexLinear(embed_dim, 1)

        mask = torch.tril(torch.ones(lmax, mmax))
        self.register_buffer("mask", mask)

        # diagonal index k = l-m
        diag_index = torch.zeros(lmax, mmax, dtype=torch.long)

        for l in range(lmax):
            for m in range(mmax):
                if l >= m:
                    diag_index[l, m] = l - m
                else:
                    diag_index[l, m] = -1

        self.register_buffer("diag_index", diag_index)

        P = torch.zeros(lmax, mmax, lmax)

        for l in range(lmax):
            for m in range(mmax):
                k = l - m
                if l >= m and 0 <= k < lmax:
                    P[l, m, k] = 1.0

        self.register_buffer("P", P)

    def get_complex_weights(self, x_spec):

        B, C, L, M = x_spec.shape
        feat = x_spec.permute(0, 2, 3, 1)
        w_raw = self.complex_project(feat).squeeze(-1)
        w_mag = w_raw.abs()
        w_mag = w_mag.masked_fill(self.mask.view(1, L, M) == 0, -1e9)
        diag_idx = self.diag_index.view(1, L, M).expand(B, -1, -1)
        w_attn = torch.zeros_like(w_mag)
        for k in range(self.lmax):
            diag_mask = (diag_idx == k)
            if diag_mask.sum() == 0:
                continue
            scores = w_mag.masked_fill(~diag_mask, -1e9)
            scores_flat = scores.view(B, -1)
            attn_flat = F.softmax(scores_flat, dim=-1)
            attn = attn_flat.view(B, L, M)
            w_attn = torch.where(diag_mask, attn, w_attn)
        w_final = w_attn * (w_raw / (w_mag + 1e-9))
        return w_final

    def slice_aggregate(self, x_spec, w):

        V = x_spec * w.conj().unsqueeze(1)
        x_contracted = torch.einsum(
            "bclm,lmk->bck",
            V,
            self.P.to(V.dtype)
        )
        return x_contracted.transpose(1, 2)

    def deslice_broadcast(self, q, w):

        Q = q.transpose(1, 2)
        Q_broad = torch.einsum(
            "bck,lmk->bclm",
            Q,
            self.P.to(Q.dtype)
        )
        x_spec_new = Q_broad * w.unsqueeze(1)

        return x_spec_new


class PhysicalMLP(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv2d(hidden_dim, dim, 1),
            nn.Dropout(drop)
        )

    def forward(self, x):
        return self.fc(x)


class DiagonalBlock(nn.Module):
    def __init__(self, ft, ift, dim, heads, mlp_r=2.0, dpr=0.):
        super().__init__()
        self.ft, self.ift = ft, ift
        self.engine = SpectralTransolverEngine(ft.lmax, ft.mmax, dim)
        
        self.q_param = nn.Parameter(torch.randn(1, ft.lmax, dim) * 0.02)
        
        self.ln_q = ComplexLayerNorm(dim)
        self.ln_s = ComplexLayerNorm(dim)
        
        self.self_attn = ComplexRealAttention(dim, heads)
        self.cross_attn = ComplexRealAttention(dim, heads)
        
        self.mlp_phys = PhysicalMLP(dim, int(dim * mlp_r))
        
        self.norm_phys = ChannelLayerNorm(dim)
        
        self.shortcut = nn.Conv2d(dim, dim, 1, bias=False)
        self.drop_path = DropPath(dpr) if dpr > 0 else nn.Identity()

    def forward(self, x):
        res = x
        B, C, H, W = x.shape
        
        with torch.autocast("cuda", enabled=False):
            x_spec = self.ft(x.float())
        
        w = self.engine.get_complex_weights(x_spec)
        x_L = self.engine.slice_aggregate(x_spec, w)
        
        q = torch.complex(self.q_param, torch.zeros_like(self.q_param)).expand(B, -1, -1)
        
        q = q + self.self_attn(self.ln_q(q), self.ln_q(q), self.ln_q(q))
        q = q + self.cross_attn(self.ln_q(q), self.ln_s(x_L), self.ln_s(x_L))
        
        x_spec_new = self.engine.deslice_broadcast(q, w)
        
        with torch.autocast("cuda", enabled=False):
            x_phys = self.ift(x_spec_new).type(res.dtype)
            
        x = self.shortcut(res) + self.drop_path(x_phys)
        x = x + self.drop_path(self.mlp_phys(self.norm_phys(x)))
        
        return x


class SpectralResample2D(nn.Module):
    def __init__(self, in_size, out_size, grid="equiangular"):
        super().__init__()
        lmax = min(in_size[0], out_size[0])
        mmax = min(in_size[1] // 2 + 1, out_size[1] // 2 + 1)
        self.ft = RealSHT(*in_size, lmax=lmax, mmax=mmax, grid=grid).float()
        self.ift = InverseRealSHT(*out_size, lmax=lmax, mmax=mmax, grid=grid).float()

    def forward(self, x):
        with torch.autocast("cuda", enabled=False):
            x_spec = self.ft(x.float())
            x_resampled = self.ift(x_spec).type(x.dtype)
        return x_resampled
    

class HybridDown2D(nn.Module):
    def __init__(self, in_size, out_size, in_channels, out_channels, grid="equiangular", normtype='bn'):
        super().__init__()
        self.pool = SpectralResample2D(in_size, out_size, grid=grid)
        self.conv = DoubleConv2D(in_channels, out_channels, normtype=normtype)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)


class HybridUp2D(nn.Module):
    def __init__(self, in_size, out_size, in_channels, out_channels, grid="equiangular", bilinear=True, normtype='bn'):
        super().__init__()
        self.up_resample = SpectralResample2D(in_size, out_size, grid=grid)
        self.bilinear = bilinear
        
        if bilinear:
            self.conv = DoubleConv2D(in_channels, out_channels, in_channels // 2, normtype=normtype)
        else:
            self.up_proj = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
            self.conv = DoubleConv2D(in_channels, out_channels, normtype=normtype)

    def forward(self, x1, x2):
        x1 = self.up_resample(x1)
        if not self.bilinear:
            x1 = self.up_proj(x1)
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class DiagNO(nn.Module):
    def __init__(self, img_size=(128, 256), grid="equiangular", scale_factor=2, 
                 in_chans=2, out_chans=2, embed_dim=128, num_heads=4, 
                 mlp_ratio=2.0, res_pred=True, dpr=0.1, bilinear=True): 
        super().__init__()
        self.res_pred = res_pred 
        
        normtype = 'in'  
        
        self.padding = [(16 - size % 16) % 16 for size in img_size]
        H_pad = img_size[0] + self.padding[0]
        W_pad = img_size[1] + self.padding[1]
        
        sizes = [
            (H_pad, W_pad),            # Level 1
            (H_pad // 2, W_pad // 2),  # Level 2
            (H_pad // 4, W_pad // 4),  # Level 3
            (H_pad // 8, W_pad // 8),  # Level 4
            (H_pad // 16, W_pad // 16) # Level 5
        ]
        
        def _get_sht_ops(size):
            modes_l = max(1, size[0] // scale_factor)
            modes_m = max(1, (size[1] // 2 + 1) // scale_factor)
            ft = RealSHT(*size, lmax=modes_l, mmax=modes_m, grid=grid).float()
            ift = InverseRealSHT(*size, lmax=modes_l, mmax=modes_m, grid=grid).float()
            return ft, ift

        self.inc = DoubleConv2D(in_chans, embed_dim, normtype=normtype)
        self.down1 = HybridDown2D(sizes[0], sizes[1], embed_dim, embed_dim * 2, grid=grid, normtype=normtype)
        self.down2 = HybridDown2D(sizes[1], sizes[2], embed_dim * 2, embed_dim * 4, grid=grid, normtype=normtype)
        self.down3 = HybridDown2D(sizes[2], sizes[3], embed_dim * 4, embed_dim * 8, grid=grid, normtype=normtype)
        factor = 2 if bilinear else 1
        self.down4 = HybridDown2D(sizes[3], sizes[4], embed_dim * 8, embed_dim * 16 // factor, grid=grid, normtype=normtype)
        
        dpr_list = [x.item() for x in torch.linspace(0, dpr, 5)] 
        
        ft1, ift1 = _get_sht_ops(sizes[0])
        self.process1 = DiagonalBlock(ft1, ift1, embed_dim, num_heads, mlp_r=mlp_ratio, dpr=dpr_list[0])
        
        ft2, ift2 = _get_sht_ops(sizes[1])
        self.process2 = DiagonalBlock(ft2, ift2, embed_dim * 2, num_heads, mlp_r=mlp_ratio, dpr=dpr_list[1])
        
        ft3, ift3 = _get_sht_ops(sizes[2])
        self.process3 = DiagonalBlock(ft3, ift3, embed_dim * 4, num_heads, mlp_r=mlp_ratio, dpr=dpr_list[2])
        
        ft4, ift4 = _get_sht_ops(sizes[3])
        self.process4 = DiagonalBlock(ft4, ift4, embed_dim * 8, num_heads, mlp_r=mlp_ratio, dpr=dpr_list[3])
        
        ft5, ift5 = _get_sht_ops(sizes[4])
        self.process5 = DiagonalBlock(ft5, ift5, embed_dim * 16 // factor, num_heads, mlp_r=mlp_ratio, dpr=dpr_list[4])
        

        self.up1 = HybridUp2D(sizes[4], sizes[3], embed_dim * 16, embed_dim * 8 // factor, grid=grid, bilinear=bilinear, normtype=normtype)
        self.up2 = HybridUp2D(sizes[3], sizes[2], embed_dim * 8, embed_dim * 4 // factor, grid=grid, bilinear=bilinear, normtype=normtype)
        self.up3 = HybridUp2D(sizes[2], sizes[1], embed_dim * 4, embed_dim * 2 // factor, grid=grid, bilinear=bilinear, normtype=normtype)
        self.up4 = HybridUp2D(sizes[1], sizes[0], embed_dim * 2, embed_dim, grid=grid, bilinear=bilinear, normtype=normtype)
        
        self.outc = OutConv2D(embed_dim, out_chans)

    def forward(self, x):
        identity = x if self.res_pred else 0
        
        if not all(item == 0 for item in self.padding):
            x = F.pad(x, [0, self.padding[1], 0, self.padding[0]])
            
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(self.process5(x5), self.process4(x4))
        x = self.up2(x, self.process3(x3))
        x = self.up3(x, self.process2(x2))
        x = self.up4(x, self.process1(x1))
        
        if not all(item == 0 for item in self.padding):
            slice_h = -self.padding[0] if self.padding[0] > 0 else None
            slice_w = -self.padding[1] if self.padding[1] > 0 else None
            x = x[..., :slice_h, :slice_w]
            
        out = self.outc(x)
        return out + identity