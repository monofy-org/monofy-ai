import os
from imageio import mimwrite
import numpy as np
import torch
import cv2
import logging
from pathlib import Path
from typing import Optional
from fastapi import BackgroundTasks, Depends
from pydantic import BaseModel
from PIL import Image
from datetime import datetime

import torchaudio
from modules.plugins import PluginBase, use_plugin, release_plugin
from plugins.video_plugin import VideoPlugin
from utils.file_utils import cached_snapshot, random_filename
from utils.gpu_utils import set_seed
from utils.image_utils import get_image_from_request
from utils.audio_utils import get_audio_from_request
from utils.stable_diffusion_utils import get_model


class Img2VidAniportraitRequest(BaseModel):
    image: str
    audio: str
    video: Optional[str] = None
    width: Optional[int] = 512
    height: Optional[int] = 512
    guidance_scale: Optional[float] = 3.5
    fps: Optional[int] = 30
    num_frames: Optional[int] = 30
    num_inference_steps: Optional[int] = 15
    seed: Optional[int] = -1


class Img2VidAniPortraitPlugin(VideoPlugin):

    name: str = "Image to Video (AniPortrait)"
    description: str = "Make images talk or sing using AniPortrait"
    instance = None

    def __init__(self):

        from omegaconf import OmegaConf
        from diffusers import (
            AutoencoderKL,
            DDIMScheduler,
        )
        from submodules.AniPortrait.src.models.unet_2d_condition import (
            UNet2DConditionModel,
        )
        from submodules.AniPortrait.src.models.unet_3d import UNet3DConditionModel
        from transformers import CLIPVisionModelWithProjection
        from submodules.AniPortrait.src.audio_models.model import Audio2MeshModel
        from submodules.AniPortrait.src.models.pose_guider import PoseGuider
        from submodules.AniPortrait.src.pipelines.pipeline_pose2vid import (
            Pose2VideoPipeline,
        )
        from submodules.AniPortrait.src.utils.frame_interpolation import (
            init_frame_interpolation_model,
        )

        super().__init__()

        config = OmegaConf.load(
            "./submodules/AniPortrait/configs/prompts/animation_audio.yaml"
        )
        if config.weight_dtype == "fp16":
            weight_dtype = torch.float16
        else:
            weight_dtype = torch.float32

        config.inference_config = (
            "./submodules/AniPortrait/configs/inference/inference_v2.yaml"
        )

        self.config = config
        self.audio_infer_config = OmegaConf.load(
            "./submodules/AniPortrait/configs/inference/inference_audio.yaml"
        )
        # prepare model
        a2m_config = self.audio_infer_config["a2m_model"]

        a2m_config.model_path = cached_snapshot("facebook/wav2vec2-base-960h")

        ckpt_path = cached_snapshot("ZJYang/AniPortrait")

        a2m_model = Audio2MeshModel(a2m_config)

        a2m_model.load_state_dict(
            torch.load(
                os.path.join(ckpt_path, "audio2mesh.pt"),
                map_location="cpu",
            ),
            strict=False,
        )
        a2m_model.cuda().eval()

        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
        ).to("cuda", dtype=weight_dtype)

        base_model_path = cached_snapshot("SG161222/Realistic_Vision_V5.1_noVAE")

        reference_unet = UNet2DConditionModel.from_pretrained(
            base_model_path,
            subfolder="unet",
        ).to(dtype=weight_dtype, device="cuda")

        inference_config_path = config.inference_config
        infer_config = OmegaConf.load(inference_config_path)
        denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            base_model_path,
            os.path.join(ckpt_path, "motion_module.pth"),
            subfolder="unet",
            unet_additional_kwargs=infer_config.unet_additional_kwargs,
        ).to(dtype=weight_dtype, device="cuda")

        pose_guider = PoseGuider(noise_latent_channels=320, use_ca=True).to(
            device="cuda", dtype=weight_dtype
        )  # not use cross attention

        image_enc = CLIPVisionModelWithProjection.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        ).to(dtype=weight_dtype, device="cuda")

        sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
        scheduler = DDIMScheduler(**sched_kwargs)

        # load pretrained weights
        denoising_unet.load_state_dict(
            torch.load(
                os.path.join(ckpt_path, "denoising_unet.pth"), map_location="cpu"
            ),
            strict=False,
        )
        reference_unet.load_state_dict(
            torch.load(
                os.path.join(ckpt_path, "reference_unet.pth"), map_location="cpu"
            ),
        )
        pose_guider.load_state_dict(
            torch.load(os.path.join(ckpt_path, "pose_guider.pth"), map_location="cpu"),
        )

        pipe = Pose2VideoPipeline(
            vae=vae,
            image_encoder=image_enc,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            pose_guider=pose_guider,
            scheduler=scheduler,
        )
        pipe = pipe.to("cuda", dtype=weight_dtype)

        # lmk_extractor = LMKExtractor()
        # vis = FaceMeshVisualizer()

        frame_inter_model = init_frame_interpolation_model(
            os.path.join(ckpt_path, "film_net_fp16.pt")
        )

        self.resources["pipe"] = pipe
        self.resources["a2m_model"] = a2m_model
        self.resources["frame_inter_model"] = frame_inter_model

    async def get_headpose_temp(self, input_video):

        from scipy.interpolate import interp1d
        from submodules.AniPortrait.src.utils.mp_utils import LMKExtractor
        from submodules.AniPortrait.src.utils.pose_util import (
            smooth_pose_seq,
            matrix_to_euler_and_translation,
        )

        lmk_extractor = LMKExtractor()
        cap = cv2.VideoCapture(input_video)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        trans_mat_list = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            result = lmk_extractor(frame)
            trans_mat_list.append(result["trans_mat"].astype(np.float32))
        cap.release()

        trans_mat_arr = np.array(trans_mat_list)

        # compute delta pose
        trans_mat_inv_frame_0 = np.linalg.inv(trans_mat_arr[0])
        pose_arr = np.zeros([trans_mat_arr.shape[0], 6])

        for i in range(pose_arr.shape[0]):
            pose_mat = trans_mat_inv_frame_0 @ trans_mat_arr[i]
            euler_angles, translation_vector = matrix_to_euler_and_translation(pose_mat)
            pose_arr[i, :3] = euler_angles
            pose_arr[i, 3:6] = translation_vector

        # interpolate to 30 fps
        new_fps = 30
        old_time = np.linspace(0, total_frames / fps, total_frames)
        new_time = np.linspace(0, total_frames / fps, int(total_frames * new_fps / fps))

        pose_arr_interp = np.zeros((len(new_time), 6))
        for i in range(6):
            interp_func = interp1d(old_time, pose_arr[:, i])
            pose_arr_interp[:, i] = interp_func(new_time)

        pose_arr_smooth = smooth_pose_seq(pose_arr_interp)

        return pose_arr_smooth

    async def generate(self, req: Img2VidAniportraitRequest):

        from submodules.AniPortrait.src.utils.mp_utils import LMKExtractor
        from submodules.AniPortrait.src.utils.draw_util import FaceMeshVisualizer
        from submodules.AniPortrait.src.utils.util import crop_face, save_videos_grid
        from submodules.AniPortrait.src.utils.audio_util import prepare_audio_feature
        from submodules.AniPortrait.src.utils.pose_util import project_points
        from submodules.AniPortrait.src.utils.frame_interpolation import (
            batch_images_interpolation_tool,
        )

        pipe = self.resources["pipe"]
        a2m_model = self.resources["a2m_model"]
        frame_inter_model = self.resources["frame_inter_model"]

        image = get_image_from_request(req.image)

        audio_path = random_filename(req.audio.split(".")[-1], False)
        audio_path = get_audio_from_request(req.audio, audio_path)

        # convert to cv2
        ref_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        fps = 30
        fi_step = 3

        lmk_extractor = LMKExtractor()
        vis = FaceMeshVisualizer()

        width = req.width
        height = req.height

        date_str = datetime.now().strftime("%Y%m%d")
        time_str = datetime.now().strftime("%H%M")

        seed, generator = set_seed(req.seed, True)

        save_dir_name = f"{time_str}--seed_{seed}-{width}x{height}"

        save_dir = Path(f"a2v_output/{date_str}/{save_dir_name}")
        while os.path.exists(save_dir):
            save_dir = Path(
                f"a2v_output/{date_str}/{save_dir_name}_{np.random.randint(10000):04d}"
            )
        save_dir.mkdir(exist_ok=True, parents=True)

        ref_image_np = cv2.cvtColor(ref_img, cv2.COLOR_RGB2BGR)
        ref_image_np = crop_face(ref_image_np, lmk_extractor)
        if ref_image_np is None:
            return None, Image.fromarray(ref_img)

        ref_image_np = cv2.resize(ref_image_np, (width, height))
        ref_image_pil = Image.fromarray(cv2.cvtColor(ref_image_np, cv2.COLOR_BGR2RGB))

        face_result = lmk_extractor(ref_image_np)
        if face_result is None:
            return None, ref_image_pil

        lmks = face_result["lmks"].astype(np.float32)
        ref_pose = vis.draw_landmarks(
            (ref_image_np.shape[1], ref_image_np.shape[0]), lmks, normed=True
        )

        sample = prepare_audio_feature(           
            audio_path,
            wav2vec_model_path=self.audio_infer_config["a2m_model"]["model_path"],
        )
        sample["audio_feature"] = (
            torch.from_numpy(sample["audio_feature"]).float().cuda()
        )
        sample["audio_feature"] = sample["audio_feature"].unsqueeze(0)

        # inference
        pred = a2m_model.infer(sample["audio_feature"], sample["seq_len"])
        pred = pred.squeeze().detach().cpu().numpy()
        pred = pred.reshape(pred.shape[0], -1, 3)
        pred = pred + face_result["lmks3d"]

        if req.video is not None:
            pose_seq = self.get_headpose_temp(req.video)
        else:
            pose_seq = np.load(self.config["pose_temp"])
        mirrored_pose_seq = np.concatenate((pose_seq, pose_seq[-2:0:-1]), axis=0)
        cycled_pose_seq = np.tile(
            mirrored_pose_seq, (sample["seq_len"] // len(mirrored_pose_seq) + 1, 1)
        )[: sample["seq_len"]]

        # project 3D mesh to 2D landmark
        projected_vertices = project_points(
            pred, face_result["trans_mat"], cycled_pose_seq, [height, width]
        )

        pose_images = []
        for i, verts in enumerate(projected_vertices):
            lmk_img = vis.draw_landmarks((width, height), verts, normed=False)
            pose_images.append(lmk_img)

        pose_list = []
        # pose_tensor_list = []

        # pose_transform = transforms.Compose(
        #     [transforms.Resize((height, width)), transforms.ToTensor()]
        # )
        args_L = (
            len(pose_images)
            if req.num_frames == 0 or req.num_frames > len(pose_images)
            else req.num_frames
        )
        args_L = min(args_L, 90)
        for pose_image_np in pose_images[:args_L:fi_step]:
            # pose_image_pil = Image.fromarray(cv2.cvtColor(pose_image_np, cv2.COLOR_BGR2RGB))
            # pose_tensor_list.append(pose_transform(pose_image_pil))
            pose_image_np = cv2.resize(pose_image_np, (width, height))
            pose_list.append(pose_image_np)

        pose_list = np.array(pose_list)

        video_length = len(pose_list)

        video = pipe(
            ref_image_pil,
            pose_list,
            ref_pose,
            width,
            height,
            video_length,
            req.num_inference_steps,
            req.guidance_scale,
            generator=generator,
        ).videos[0]

        print(f"Video: {video}")

        # video = batch_images_interpolation_tool(
        #     video, frame_inter_model, inter_frames=fi_step - 1
        # )
        
        frames = []
        for frame in video:
            frames.append(Image.fromarray(frame))        

        return frames, audio_path


@PluginBase.router.post("/img2vid/aniportrait", tags=["Image-to-Video"])
async def img2vid_aniportrait(
    background_tasks: BackgroundTasks, req: Img2VidAniportraitRequest
):
    plugin: Img2VidAniPortraitPlugin = None
    try:
        plugin = await use_plugin(Img2VidAniPortraitPlugin)
        frames, audio = await plugin.generate(req)

        print(f"Frames: {frames}")
        print(f"Audio: {audio}")

        return plugin.video_response(background_tasks, frames, audio=audio, fps=req.fps)

    except Exception as e:
        logging.error(f"Error in img2vid_aniportrait: {e}", exc_info=True)
        return {"error": str(e)}

    finally:
        if plugin:
            release_plugin(plugin)


@PluginBase.router.get("/img2vid/aniportrait", tags=["Image-to-Video"])
async def img2vid_aniportrait_from_url(req: Img2VidAniportraitRequest = Depends()):
    return await img2vid_aniportrait(req)
