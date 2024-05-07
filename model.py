import torch.nn as nn
import torch
import math


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, theta: int = 10000) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        temb = math.log(self.theta) / (half_dim - 1)
        temb = torch.exp(torch.arange(half_dim, device=device) * -temb)
        temb = x[:, None] * temb[None, :]
        temb = torch.cat((temb.sin(), temb.cos()), dim=-1)
        return temb
    

class Downsample(nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()

        self.downsample = nn.Conv2d(dim_in, dim_out, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(x)


class Upsample(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, up_scale:int = 2) -> None:
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=up_scale, mode='nearest'),
            nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(x)


class Conv2dBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, groups: int = 8) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, 1),
            nn.GroupNorm(groups, dim_out),
            nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Resblock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, dim_temb, groups: int = 8) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_temb, dim_out)
        )

        self.conv_block1 = Conv2dBlock(dim_in, dim_out, groups=groups)
        self.conv_block2 = Conv2dBlock(dim_out, dim_out, groups=groups)
        self.shortcut_conv = Conv2dBlock(dim_in, dim_out, groups=1) if dim_in != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        temb = self.mlp(temb)
        s = self.conv_block1(x)
        s = s + temb[:, :, None, None]
        s = self.conv_block2(s)
        return s + self.shortcut_conv(x)


class AttenBlock(nn.Module):
    def __init__(self, dim: int, dim_head: int, heads: int = 4, groups: int = 8) -> None:
        super().__init__()

        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads

        self.norm = nn.GroupNorm(groups, dim)

        self.to_qkv = nn.Conv2d(dim, dim_head * 3 * heads, 1, 1)
        self.to_out = nn.Conv2d(dim_head * heads, dim, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        n, c, h, w = x.shape
        
        x = self.norm(x)

        qkv = self.to_qkv(x)
        qkv = qkv.view(n, self.heads, self.dim_head * 3, -1)

        q, k, v = torch.chunk(qkv, chunks=3, dim=2)  # split tensor along with c

        q = q.view(n, self.heads, -1, self.dim_head)
        k = k.view(n, self.heads, -1, self.dim_head)
        v = v.view(n, self.heads, -1, self.dim_head)

        score = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.dim)
        weight = torch.softmax(score, dim=2)
        atten = torch.matmul(weight, v)

        atten = atten.permute(0, 1, 3, 2).contiguous()
        atten = atten.view(n, -1, h, w)

        return x + self.to_out(atten)


class Unet(nn.Module):
    def __init__(
            self, 
            dim_in: int = 3,
            init_dim: int = 64,
            dim_out: int = 3,
            dim_temb: int = 512,
            heads: int = 4,
            groups: int = 8) -> None:
        
        super().__init__()

        self.init_conv2d = nn.Conv2d(dim_in, init_dim, 3, 1, 1)

        self.dims = [init_dim, init_dim * 2, init_dim * 4, init_dim * 8]
        self.resolutions = [32, 16, 8, 4]

        self.shared_temb_mlp = nn.Sequential(
            SinusoidalPosEmb(dim_temb),
            nn.Linear(dim_temb, dim_temb >> 2),
            nn.GELU(),
            nn.Linear(dim_temb >> 2, dim_temb >> 2)
        )

        self.downsamples = nn.ModuleList([])
        self.middles = nn.ModuleList([])
        self.upsamples = nn.ModuleList([])

        # make downsample layers
        for i in [0, 1, 2, 3]:
            self.downsamples.append(
                nn.ModuleList([
                    Resblock(self.dims[i], self.dims[i], dim_temb >> 2, groups=groups),
                    Resblock(self.dims[i], self.dims[i], dim_temb >> 2, groups=groups),
                    AttenBlock(self.dims[i], self.dims[i] // heads, heads, groups=groups),
                    Downsample(self.dims[i], self.dims[i + 1]) if i != 3 else nn.Conv2d(self.dims[i], self.dims[i], 3, 1, 1)
                ])
            )

        # last 4*4 init_dim * 8
        
        # make middle layers
        self.middles.append(nn.ModuleList([
            Resblock(self.dims[-1], self.dims[-1], dim_temb >> 2, groups=groups),
            AttenBlock(self.dims[-1], self.dims[-1] // heads, heads, groups=groups),
            Resblock(self.dims[-1], self.dims[-1], dim_temb >> 2, groups=groups)
        ]))

        # make upsample layers
        for i in [3, 2, 1, 0]:
            self.upsamples.append(
                nn.ModuleList([
                    Resblock(self.dims[i] * 2, self.dims[i], dim_temb >> 2, groups=groups),
                    Resblock(self.dims[i] * 2, self.dims[i], dim_temb >> 2, groups=groups),
                    AttenBlock(self.dims[i], self.dims[i] // heads, heads, groups=groups),
                    Upsample(self.dims[i], self.dims[i - 1]) if i != 0 else nn.Conv2d(self.dims[i], self.dims[i], 3, 1, 1)
                ])
            )
        
        self.final_res = Resblock(self.dims[0], self.dims[0], dim_temb >> 2, groups=groups)
        self.final_conv2d = nn.Conv2d(self.dims[0], dim_out, 3, 1, 1)


    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.init_conv2d(x)
        temb = self.shared_temb_mlp(t)

        shortcut = []

        for (resblock1, resblock2, attenblock, downsample) in self.downsamples:
            x = resblock1(x, temb)
            shortcut.append(x)

            x = resblock2(x, temb)
            shortcut.append(x)

            x = attenblock(x)
            x = downsample(x)
        
        for (resblock1, attenblock, resblock2) in self.middles:
            x = resblock1(x, temb)
            x = attenblock(x)
            x = resblock2(x, temb)
            

        for (resblock1, resblock2, attenblock, upsample) in self.upsamples:
            x = torch.cat((x, shortcut.pop()), dim=1)
            x = resblock1(x, temb)
            x = torch.cat((x, shortcut.pop()), dim=1)
            x = resblock2(x, temb)
            x = attenblock(x)
            x = upsample(x)
        
        x = self.final_res(x, temb)
        return self.final_conv2d(x)


