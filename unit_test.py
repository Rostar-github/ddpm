import torch
from model import AttenBlock, SinusoidalPosEmb, Unet, Resblock


if __name__ == "__main__":
    block = Unet(dim_in=3, init_dim=64, dim_out=3, dim_temb=256, heads=4, groups=8).cuda()
    x = torch.randn((8, 3, 32, 32)).cuda()
    t = torch.tensor([2, 4, 6, 19, 20, 88, 102, 998], dtype=torch.float32).cuda()
    print(block(x, t).shape)

    # block1 = Resblock(64, 64, 32).cuda()
    # block2 = SinusoidalPosEmb(32).cuda()
    # x = torch.randn((8, 64, 32, 32)).cuda()
    # t = torch.tensor([2, 4, 6, 19, 20, 88, 102, 998], dtype=torch.float32).cuda()
    # print(block1(x, block2(t)).shape)
