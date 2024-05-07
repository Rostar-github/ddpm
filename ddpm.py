import torch
import os
import argparse
from torchvision.datasets import CIFAR10
from torchvision.transforms import Resize, RandomHorizontalFlip, ToTensor, Normalize, Compose
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from model import Unet
from utils import get_batch_timesteps, get_batch_acc_alphas

from tqdm import tqdm


def set_env():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

def setup(rank, world_size):
    torch.cuda.set_device(rank)
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def train(args, rank, epoch, dataloader, model, optimizer, mse, acc_alphas):
    model.train()
    for idx, (img, _) in tqdm(enumerate(dataloader)):
        img = img.cuda()
        z = torch.randn_like(img).cuda()

        t = get_batch_timesteps(img.shape[0], args.iters).cuda()
        acc_alpha_t = get_batch_acc_alphas(acc_alphas, t - 1).cuda()

        acc_alpha_t = acc_alpha_t[:, None, None, None]  # unsquezze

        x_t = torch.sqrt(acc_alpha_t) * img + torch.sqrt(1 - acc_alpha_t) * z

        model.zero_grad()

        pred_z = model(x_t, t)
        loss = mse(pred_z, z)
        loss.backward()

        optimizer.step()

    print(f"Rank {rank} Epoch {epoch + 1} MSE: {loss:<0.4f}")


def sample_ddpm(args, rank, epoch, model, alphas, acc_alphas):
    print(f"Rank {rank} Sampling...")
    model.eval()
    with torch.no_grad():
        x_t = torch.randn((args.batch_size // 2, 3, args.size, args.size)).cuda()
        for t in tqdm(reversed(range(1, args.iters + 1))):
            t_tensor = torch.tensor([t]).cuda()
            # Reparameterization sample
            if not t == 1:
                z = torch.randn((args.batch_size // 2, 3, args.size, args.size)).cuda()
            else: 
                z = 0
            x_t = (x_t - (1 - alphas[t-1]) * model(x_t, t_tensor) / torch.sqrt(1-acc_alphas[t-1])) / torch.sqrt(alphas[t-1]) + \
            z * torch.sqrt(1 - alphas[t-1])

    if rank == 0:
        gimg = x_t
        grid = make_grid(gimg, nrow=8, padding=2, normalize=True)
        save_image(grid, f"imgs/ddpm/{epoch + 1}.jpg")


def sample_ddim(args, rank, epoch, model, alphas, acc_alphas):
    print(f"Rank {rank} Sampling...")
    model.eval()
    c = 10 
    # subsequence = {10, 20, 30, ..., 100}
    with torch.no_grad():
        x_t = torch.randn((args.batch_size // 2, 3, args.size, args.size)).cuda()
        for t in tqdm(reversed(range(2, args.iters // c + 1))):
            t_next = (t - 1) * c
            t = t * c
            t_tensor = torch.tensor([t]).cuda()
            # Deterministic sample
            pred_noise = model(x_t, t_tensor)
            coef_acc_alphas_divide = torch.sqrt(acc_alphas[t_next-1] / acc_alphas[t-1])
            coef_pred_noise_next_t = torch.sqrt(1 - acc_alphas[t_next-1])
            coef_pred_noise_t = torch.sqrt(1 - acc_alphas[t-1])
            x_t = coef_acc_alphas_divide * (x_t - coef_pred_noise_t * pred_noise) + coef_pred_noise_next_t * pred_noise

    if rank == 0:
        gimg = x_t
        grid = make_grid(gimg, nrow=8, padding=2, normalize=True)
        save_image(grid, f"imgs/ddim/{epoch + 1}.jpg")


def main(rank, args):

    setup(rank, world_size=args.ngpu)
    
    augment = Compose([Resize(32), 
                       RandomHorizontalFlip(), 
                       ToTensor(), 
                       Normalize(mean=0.5, std=0.5)])
    
    dataset = CIFAR10(root="./dataset/cifar10", 
                      train=True, 
                      download=False, 
                      transform=augment)
    
    sampler = DistributedSampler(dataset=dataset, shuffle=True, rank=rank)
    
    dataloader = DataLoader(dataset=dataset, 
                            batch_size=args.batch_size, 
                            sampler=sampler,
                            pin_memory=True, 
                            num_workers=8)

    unet = Unet().cuda()
    ddp_unet = DDP(unet, device_ids=[rank])

    optimizer = torch.optim.Adam(ddp_unet.parameters(), lr=args.lr)

    mse = torch.nn.MSELoss()

    betas = torch.linspace(start=args.beta_min, end=args.beta_max, steps=args.iters).cuda()
    alphas = 1 - betas
    acc_alphas = torch.cumprod(alphas, dim=0).cuda()

    # sample_ddpm(args, 0, unet, alphas, acc_alphas) # unit test

    for epoch in range(args.epochs):
        train(args, rank, epoch, dataloader, ddp_unet, optimizer, mse, acc_alphas)
        if (epoch + 1) % args.eval_interval == 0:
            if args.sample_mode == "ddpm":
                sample_ddpm(args, rank, epoch, ddp_unet, alphas, acc_alphas)
            if args.sample_mode == "ddim":
                sample_ddim(args, rank, epoch, ddp_unet, alphas, acc_alphas)
                
    if rank == 0:
        torch.save(unet, "checkpoints/unet_ddim.pth")


def run_ddp(args):
    mp.spawn(fn=main, args=(args,), nprocs=args.ngpu, join=True)


if __name__ == "__main__":
    set_env()
    parser = argparse.ArgumentParser()

    parser.add_argument("--sample_mode", type=str, default="ddim", choices=["ddpm", "ddim"], help="Type of generation sample")

    parser.add_argument("--epochs", type=int, default=200, help="training epochs")
    parser.add_argument("--iters", type=int, default=1000, help="number of iterations of adding noise step")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    parser.add_argument("--size", type=int, default=32, help="image size")
    parser.add_argument("--beta_min", type=float, default=1e-4, help="minimal beta")
    parser.add_argument("--beta_max", type=float, default=0.02, help="maximal beta")
    parser.add_argument("--eval_interval", type=int, default=10, help="perform evaluating per x epochs")

    parser.add_argument("--ngpu", type=int, default=2, help="number of GPUs")

    args = parser.parse_args()

    run_ddp(args)
