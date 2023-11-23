import torch.nn as nn
import torch
import math
from config import FLAGES

class matmul(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x = x1 @ x2
        return x

def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initialy created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=gelu):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale =head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.mat = matmul()

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (self.mat(q, k.transpose(-2, -1))) * self.scale

        attn = attn.softmax(dim=-1)

        x = self.mat(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., act_layer=gelu, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Transpan(nn.Module): #  cited from the paper "Vision Transformer for Pansharpening" in 2022 TGRS
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(FLAGES.num_spectrum+1, FLAGES.embed_dim, kernel_size=FLAGES.patch_size, stride=FLAGES.patch_size, padding=0)
        num_patches = (FLAGES.pan_size //FLAGES.patch_size) ** 2

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, FLAGES.embed_dim))

        self.blocks = nn.ModuleList([
            Block(dim=FLAGES.embed_dim, num_heads=FLAGES.num_heads, mlp_ratio=4,norm_layer=nn.LayerNorm) for _ in range(FLAGES.depth)])
        self.norm = nn.LayerNorm(FLAGES.embed_dim)
        self.deconv = nn.Sequential(
            nn.Conv2d(FLAGES.embed_dim, FLAGES.num_spectrum*FLAGES.patch_size*FLAGES.patch_size, 1, 1, 0)
        )
        self.relu = nn.ReLU()
        self.ps4=nn.PixelShuffle(FLAGES.patch_size)
        self.upsample=nn.Upsample(scale_factor=(4,4),mode='bicubic',align_corners=True)

    def forward(self, ms,pan):
        up_ms=self.upsample(ms)
        x = self.patch_embed(torch.cat((up_ms,pan),1)).flatten(2).permute(0, 2, 1)

        x=x+self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        N = int(math.sqrt(x.shape[2]))
        x = x.view(-1, FLAGES.embed_dim, N, N)
        x = self.deconv(x)
        x = self.ps4(x)
        return x