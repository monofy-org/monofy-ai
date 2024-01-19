#!/bin/env python

import _thread
import logging
import time
from queue import Queue
import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from tqdm.rich import tqdm
from modules.rife.ssim import ssim_matlab
from modules.rife.model_rife import RifeModel
from modules import devices, shared

# Thanks to vladmandic for this implementation https://github.com/vladmandic/
# Shoutout to the SD.Next team for their help and encouragement.
# Don't blame them if I broke anything.

model: RifeModel = None

def load(model_path: str = 'modules/rife/flownet-v46.pkl'):
    global model # pylint: disable=global-statement
    if model is None:                        
        logging.info(f'RIFE load model: file="{model_path}"')
        model = RifeModel()
        model.load_model(model_path, -1)
        model.eval()
        model.device()


def interpolate(images: list, count: int = 2, scale: float = 1.0, pad: int = 1, change: float = 0.3):
    if images is None or len(images) < 2:
        return []
    if model is None:
        load()
    interpolated = []
    is_images = False
    try:
        first = Image.fromarray(images[0])
    except Exception:
        is_images = True
        first = images[0]
    
    h = first.height
    w = first.width
    t0 = time.time()

    def write(buffer):
        item = buffer.get()
        while item is not None:
            img = item[:, :, ::-1]
            # image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            image = Image.fromarray(img)
            item = buffer.get()
            interpolated.append(image)

    def execute(I0, I1, n):
        if model.version >= 3.9:
            res = []
            for i in range(n):
                res.append(model.inference(I0, I1, (i+1) * 1. / (n+1), scale))
            return res
        else:
            middle = model.inference(I0, I1, scale)
            if n == 1:
                return [middle]
            first_half = execute(I0, middle, n=n//2)
            second_half = execute(middle, I1, n=n//2)
            if n % 2:
                return [*first_half, middle, *second_half]
            else:
                return [*first_half, *second_half]

    def f_pad(img):
        return F.pad(img, padding).to(devices.dtype) # pylint: disable=not-callable

    tmp = max(128, int(128 / scale))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)
    buffer = Queue(maxsize=8192)
    _thread.start_new_thread(write, (buffer,))

    frame = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
    for _i in range(pad): # fill starting frames
        buffer.put(frame)

    I1 = f_pad(torch.from_numpy(np.transpose(frame, (2,0,1))).to(devices.device, non_blocking=True).unsqueeze(0).float() / 255.)
    with torch.no_grad():
        with tqdm(total=len(images), desc='Interpolate', unit='frame') as pbar:
            for image in images:
                try:
                    frame = cv2.cvtColor(np.array(image) if is_images else image, cv2.COLOR_RGB2BGR)
                    I0 = I1
                    I1 = f_pad(torch.from_numpy(np.transpose(frame, (2,0,1))).to(devices.device, non_blocking=True).unsqueeze(0).float() / 255.)
                    I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False).to(torch.float32)
                    I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False).to(torch.float32)
                    ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
                    if ssim > 0.999: # skip duplicate frames
                        pbar.update(1)
                        continue
                    if ssim < change:
                        output = []
                        for _i in range(pad): # fill frames if change rate is above threshold
                            output.append(I0)
                        for _i in range(pad):
                            output.append(I1)
                    else:
                        output = execute(I0, I1, count-1)
                    for mid in output:
                        mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
                        buffer.put(mid[:h, :w])
                    buffer.put(frame)
                    pbar.update(1)
                except Exception as e:
                    logging.error(f'RIFE interpolate: error={e}')                    
                    shared.error = True
                    break
    for _i in range(pad): # fill ending frames
        buffer.put(frame)
    while not buffer.empty():
        time.sleep(0.1)
    t1 = time.time()
    logging.info(f'RIFE interpolate: input={len(images)} frames={len(interpolated)} resolution={w}x{h} interpolate={count} scale={scale} pad={pad} change={change} time={round(t1 - t0, 2)}')
    return interpolated
