import torch


def get_batch_timesteps(batch_size: int, iters: int) -> torch.Tensor:
    timestep = torch.rand(batch_size) * iters + 1
    return timestep.type(torch.long)

def get_batch_acc_alphas(acc_alphas: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    return acc_alphas[timesteps]