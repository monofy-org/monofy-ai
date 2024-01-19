import torch
from utils.gpu_utils import autodetect_device, use_fp16


device = torch.device(autodetect_device())
dtype = torch.float16 if use_fp16 else torch.float32