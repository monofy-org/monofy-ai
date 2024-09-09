import torch
from utils.gpu_utils import autodetect_device, autodetect_dtype


device = autodetect_device()
dtype = torch.float32 # autodetect_dtype()