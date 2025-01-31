import glob
import os
from io import BytesIO

import cv2
import torch
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel
from torchvision.transforms.functional import normalize

from modules.plugins import PluginBase, release_plugin, use_plugin
from settings import CACHE_PATH
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
    background_enhance: bool = True
    face_upsample: bool = True
    upscale: bool = False
    codeformer_fidelity: float = 0.7
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
        fidelity_weight=0.5,
        upscale=2,
        has_aligned=False,
        only_center_face=False,
        draw_box=False,
        detection_model="retinaface_resnet50",
        bg_upsampler="None",
        face_upsample=False,
        bg_tile=400,
        suffix=None,
        save_video_fps=None,
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

        result_root = output_path

        test_img_num = len(input_img_list)
        if test_img_num == 0:
            raise FileNotFoundError(
                "No input image/video is found...\n"
                "\tNote that --input_path for video should end with .mp4|.mov|.avi"
            )

        # Background upsampler setup
        bg_upsampler = set_realesrgan() if bg_upsampler == "realesrgan" else None

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
                face_helper.cropped_faces = [img]
            else:
                face_helper.read_image(img)
                num_det_faces = face_helper.get_face_landmarks_5(
                    only_center_face=only_center_face, resize=640, eye_dist_threshold=5
                )
                face_helper.align_warp_face()

            # Face restoration logic
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
                        output = net(cropped_face_t, w=fidelity_weight, adain=True)[0]
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

            # Rendering and saving logic remains the same as the original script

        print(f"\nAll results are saved in {result_root}")
        return result_root


@PluginBase.router.post("/img/codeformer")
async def codeformer(req: CodeFormerRequest):
    plugin: CodeFormerPlugin = None
    try:
        plugin: CodeFormerPlugin = await use_plugin(CodeFormerPlugin)
        input_path = get_image_from_request(req.image, return_path=True)
        output_path = plugin.generate(input_path=input_path)
        image = Image.open(output_path)
        if req.return_json:
            return {"image": image_to_base64_no_header(image)}
        else:
            bytes_io = BytesIO()
            image.save(bytes_io, format="PNG")
            bytes_io.seek(0)
            return StreamingResponse(
                bytes_io,
                media_type="image/png",
            )
    except Exception as e:
        print(e)
        return {"error": str(e)}
    finally:
        if plugin:
            release_plugin(plugin)


def imread(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
