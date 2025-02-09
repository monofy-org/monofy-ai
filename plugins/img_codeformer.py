import glob
import logging
import os

import cv2
import torch
from fastapi import Depends
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel
from torchvision.transforms.functional import normalize

from modules.plugins import PluginBase, release_plugin, use_plugin
from submodules.CodeFormer.inference_codeformer import (
    ARCH_REGISTRY,
    FaceRestoreHelper,
    img2tensor,
    is_gray,
    load_file_from_url,
    set_realesrgan,
    tensor2img,
)
from utils.file_utils import random_filename
from utils.image_utils import get_image_from_request, image_to_base64_no_header


class CodeFormerRequest(BaseModel):
    image: str
    face_align: bool = True
    background_enhance: bool = False
    face_upsample: bool = True
    upscale: float = 1.0
    strength: float = 0.5
    return_json: bool = False


class CodeFormerPlugin(PluginBase):
    name = "Img2Img CodeFormer"
    description = "Image restoration with CodeFormer"
    instance = None

    def __init__(self):
        super().__init__()

    def generate(
        self,
        input_path: str,
        output_path: str = random_filename("png"),
        strength: float = 0.5,
        upscale: float = 1.0,
        has_aligned: bool = False,
        only_center_face: bool = False,
        draw_box: bool = False,
        detection_model: str = "retinaface_resnet50",
        background_enhance: bool = False,
        face_upsample: bool = False,
        bg_tile: int = 400,
        save_video_fps: int = None,
    ):
        device = self.device

        # Input & output handling
        input_video = False
        if input_path.endswith(("jpg", "jpeg", "png", "JPG", "JPEG", "PNG")):
            input_img_list = [input_path]
        elif input_path.endswith(("mp4", "mov", "avi", "MP4", "MOV", "AVI")):
            from submodules.CodeFormer.basicsr.utils.video_util import (
                VideoReader,
                VideoWriter,
            )

            input_img_list = []
            vidreader = VideoReader(input_path)
            image = vidreader.get_frame()
            while image is not None:
                input_img_list.append(image)
                image = vidreader.get_frame()
            audio = vidreader.get_audio()
            fps = vidreader.get_fps() if save_video_fps is None else save_video_fps
            video_name = os.path.basename(input_path)[:-4]
            input_video = True
            vidreader.close()
        else:
            if input_path.endswith("/"):
                input_path = input_path[:-1]
            input_img_list = sorted(
                glob.glob(os.path.join(input_path, "*.[jpJP][pnPN]*[gG]"))
            )

        test_img_num = len(input_img_list)
        if test_img_num == 0:
            raise FileNotFoundError(
                "No input image/video is found...\n"
                "\tNote that --input_path for video should end with .mp4|.mov|.avi"
            )

        # Background upsampler setup
        bg_upsampler = set_realesrgan() if background_enhance else None

        # Face upsampler setup
        face_upsampler = bg_upsampler if face_upsample else None

        # CodeFormer model setup
        net = ARCH_REGISTRY.get("CodeFormer")(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).to(device)

        ckpt_path = load_file_from_url(
            url="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
            model_dir="models/CodeFormer",
            progress=True,
            file_name=None,
        )
        checkpoint = torch.load(ckpt_path)["params_ema"]
        net.load_state_dict(checkpoint)
        net.eval()

        # Face restore helper setup
        face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model=detection_model,
            save_ext="png",
            use_parse=True,
            device=device,
        )

        # Processing
        for i, img_path in enumerate(input_img_list):
            face_helper.clean_all()

            if isinstance(img_path, str):
                img_name = os.path.basename(img_path)
                basename, ext = os.path.splitext(img_name)
                print(f"[{i+1}/{test_img_num}] Processing: {img_name}")
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            else:
                basename = str(i).zfill(6)
                img_name = f"{video_name}_{basename}" if input_video else basename
                print(f"[{i+1}/{test_img_num}] Processing: {img_name}")
                img = img_path

            if has_aligned:
                img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
                face_helper.is_gray = is_gray(img, threshold=10)
                if face_helper.is_gray:
                    print("Grayscale input: True")
                face_helper.cropped_faces = [img]
            else:
                face_helper.read_image(img)
                num_det_faces = face_helper.get_face_landmarks_5(
                    only_center_face=only_center_face, resize=640, eye_dist_threshold=5
                )
                print(f"\tdetect {num_det_faces} faces")
                face_helper.align_warp_face()

            for idx, cropped_face in enumerate(face_helper.cropped_faces):
                cropped_face_t = img2tensor(
                    cropped_face / 255.0, bgr2rgb=True, float32=True
                )
                normalize(
                    cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True
                )
                cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

                try:
                    with torch.no_grad():
                        output = net(cropped_face_t, w=strength, adain=True)[0]
                        restored_face = tensor2img(
                            output, rgb2bgr=True, min_max=(-1, 1)
                        )
                    del output
                    torch.cuda.empty_cache()
                except Exception as error:
                    print(f"\tFailed inference for CodeFormer: {error}")
                    restored_face = tensor2img(
                        cropped_face_t, rgb2bgr=True, min_max=(-1, 1)
                    )

                restored_face = restored_face.astype("uint8")
                face_helper.add_restored_face(restored_face, cropped_face)

            if not has_aligned:
                if bg_upsampler is not None:
                    bg_img = bg_upsampler.enhance(img, outscale=upscale)[0]
                else:
                    bg_img = None
                face_helper.get_inverse_affine(None)

                if face_upsample and face_upsampler is not None:
                    restored_img = face_helper.paste_faces_to_input_image(
                        upsample_img=bg_img,
                        draw_box=draw_box,
                        face_upsampler=face_upsampler,
                    )
                else:
                    restored_img = face_helper.paste_faces_to_input_image(
                        upsample_img=bg_img, draw_box=draw_box
                    )

        cv2.imwrite(output_path, restored_img)
        return output_path


@PluginBase.router.post("/img/codeformer")
async def codeformer(req: CodeFormerRequest):
    plugin: CodeFormerPlugin = None
    try:
        plugin: CodeFormerPlugin = await use_plugin(CodeFormerPlugin)
        input_path = get_image_from_request(req.image, return_path=True)
        output_path = plugin.generate(
            input_path=input_path,
            background_enhance=req.background_enhance,
            upscale=req.upscale,
            strength=req.strength,
        )
        if req.return_json:
            image = Image.open(output_path)
            return {"images": [image_to_base64_no_header(image)]}
        else:
            return FileResponse(output_path, media_type="image/png")
    except Exception as e:
        logging.error(e, exc_info=True)
        return {"error": str(e)}
    finally:
        if plugin:
            release_plugin(plugin)


@PluginBase.router.get("/img/codeformer")
async def codeformer_get(req: CodeFormerRequest = Depends()):
    return await codeformer(req)


def imread(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
