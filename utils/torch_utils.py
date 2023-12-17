import torch


def autodetect_device():
    """Returns a device such as "cpu" or "cuda:0" """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_seed(value=-1):
    """Returns a seed for deterministic results (defaults to -1 for a random seed)"""
    if value == -1:
        return torch.manual_seed(torch.randint(0, 2**32, (1,)).item())
    else:
        return torch.manual_seed(value)
