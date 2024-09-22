from fastapi import Depends
from pydantic import BaseModel
from modules.plugins import PluginBase, release_plugin, use_plugin
from plugins.experimental.vid2vid_frames import get_video_from_request
from PIL import Image


class Vid2TxtRequest(BaseModel):
    video: str    
    num_frames: int


class Vid2TxtVideoMAEPlugin(PluginBase):
    name = "Vid2TxtVideoMAE"
    description = "Video to text using VideoMAE"
    instance = None

    def __init__(self):
        super().__init__()

        from transformers import VideoMAEImageProcessor, VideoMAEForPreTraining

        self.resources["processor"] = VideoMAEImageProcessor.from_pretrained(
            "MCG-NJU/videomae-base"
        )
        self.resources["model"] = VideoMAEForPreTraining.from_pretrained(
            "MCG-NJU/videomae-base"
        )

    async def generate(self, req: Vid2TxtRequest):

        import torch        
        from transformers import VideoMAEImageProcessor, VideoMAEForPreTraining
        from torchvision.io import read_video

        processor: VideoMAEImageProcessor = self.resources["processor"]
        model: VideoMAEForPreTraining = self.resources["model"]

        video_path = get_video_from_request(req.video)
        video = read_video(video_path)

        # the next line processor(... throws the following:
        # ValueError: Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, torch.Tensor, tf.Tensor or jax.ndarray

        # Convert video to frames
        
        frames = []
        for frame in video[0][0][0]:
            frames.append(frame.permute(1, 2, 0).numpy())
        
        pixel_values = processor(frames, return_tensors="pt").pixel_values

        num_patches_per_frame = (
            model.config.image_size // model.config.patch_size
        ) ** 2
        seq_length = (
            req.num_frames // model.config.tubelet_size
        ) * num_patches_per_frame
        bool_masked_pos = torch.randint(0, 2, (1, seq_length)).bool()

        outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
        loss = outputs.loss

        return outputs, loss


@PluginBase.router.post("/vid2txt/videomae")
async def vid2txt_videomae(req: Vid2TxtRequest):
    plugin: Vid2TxtVideoMAEPlugin = None
    try:
        plugin = await use_plugin(Vid2TxtVideoMAEPlugin)
        return await plugin.generate(req)

    finally:
        if plugin:
            release_plugin(plugin)


@PluginBase.router.get("/vid2txt/videomae")
async def vid2txt_videomae_from_url(req: Vid2TxtRequest = Depends()):
    return await vid2txt_videomae(req)
