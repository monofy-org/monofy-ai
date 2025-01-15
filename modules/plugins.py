import torch
import gc
import logging
import time
from typing import Type
from fastapi.routing import APIRouter
from asyncio import Lock

from settings import KEEP_FLUX_LOADED
from utils.console_logging import Emojis, log_plugin, log_recycle
from utils.gpu_utils import (
    autodetect_device,
    autodetect_dtype,
    bytes_to_gib,
    clear_gpu_cache,
)


def load_plugins():
    from plugins.stable_diffusion import StableDiffusionPlugin
    from plugins.txt2img_canny import Txt2ImgCannyPlugin
    from plugins.txt2img_depth import Txt2ImgDepthMidasPlugin
    from plugins.txt2img_openpose import Txt2ImgOpenPosePlugin

    # from plugins.experimental.txt2img_instantid import Txt2ImgInstantIDPlugin
    from plugins.txt2img_cascade import Txt2ImgCascadePlugin
    from plugins.txt2img_controlnet import Txt2ImgControlNetPlugin
    from plugins.txt2img_flux import Txt2ImgFluxPlugin

    # from plugins.experimental.txt2img_photomaker import Txt2ImgPhotoMakerPlugin
    from plugins.txt2img_relight import Txt2ImgRelightPlugin
    from plugins.extras.txt2img_zoom import Txt2ImgZoomPlugin

    from plugins.txt2vid_animate import Txt2VidAnimatePlugin

    # from plugins.experimental.txt2vid_cogvideox import Txt2VidCogVideoXPlugin
    from plugins.txt2vid_zeroscope import Txt2VidZeroscopePlugin
    from plugins.experimental.img2vid_genxl import Img2VidGenXLPlugin
    from plugins.img2vid_xt import Img2VidXTPlugin
    from plugins.img2vid_liveportrait import Img2VidLivePortraitPlugin
    from plugins.vid2vid_latentsync import Vid2VidLatentSyncPlugin
    from plugins.experimental.vid2vid_magicanimate import Vid2VidMagicAnimatePlugin
    from plugins.experimental.img2vid_aniportrait import Img2VidAniPortraitPlugin
    from plugins.txt2vid import Txt2VidZeroPlugin
    from plugins.txt2wav_stable_audio import Txt2WavStableAudioPlugin
    from plugins.img_depth_anything import DepthAnythingPlugin
    from plugins.img_depth_midas import DepthMidasPlugin
    from plugins.detect_yolos import DetectYOLOSPlugin
    from plugins.detetct_owl import DetectOwlPlugin
    from plugins.img2model_lgm import Img2ModelLGMPlugin
    from plugins.img2model_tsr import Img2ModelTSRPlugin

    # from plugins.experimental.img2model_vfusion import Img2ModelVFusionPlugin
    # from plugins.experimental.img2model_stablefast3d import Img2ModelStableFast3DPlugin
    from plugins.img_rembg import RembgPlugin
    from plugins.experimental.img_upres import ImgUpresPlugin
    from plugins.img2txt_moondream import Img2TxtMoondreamPlugin
    from plugins.img2txt_llava import Img2TxtLlavaPlugin
    from plugins.txt2wav_musicgen import Txt2WavMusicGenPlugin
    from plugins.exllamav2 import ExllamaV2Plugin
    from plugins.mmaudio import MMAudioPlugin
    from plugins.experimental.causal_lm import CausalLMPlugin
    from plugins.txt2model_shap_e import Txt2ModelShapEPlugin
    from plugins.txt2model_avatar import Txt2ModelAvatarPlugin
    from plugins.tts import TTSPlugin
    from plugins.txt_summary import TxtSummaryPlugin
    from plugins.voice_whisper import VoiceWhisperPlugin
    from plugins.voice_conversation import VoiceConversationPlugin
    from plugins.experimental.vid2vid_frames import Vid2VidPlugin
    from plugins.vid2densepose import Vid2DensePosePlugin
    # from plugins.video_ai import VideoAIPlugin
    # import plugins.img_openpose
    import plugins.extras.unload
    import plugins.extras.tts_edge
    import plugins.extras.txt_profile
    import plugins.extras.txt2img_face
    import plugins.extras.img_canny
    import plugins.extras.img_exif
    # import plugins.extras.video_crop
    import plugins.extras.pdf_rip
    import plugins.extras.reddit
    import plugins.extras.youtube
    import plugins.extras.video_download_m3u8
    import plugins.extras.file_share
    import plugins.extras.google_trends
    import plugins.extras.wav_demucs
    import plugins.extras.piano2midi
    import plugins.extras.geo_zip2coords

    # from plugins.experimental.txt2img_pano360 import Txt2ImgPano360Plugin
    # from plugins.experimental.txt2img_pulid import Txt2ImgPuLIDPlugin
    # from plugins.experimental.txt2vid_animatediff import Txt2VidAnimateDiffPlugin
    # from plugins.experimental.txt2vid_vader import Txt2VidVADERPlugin
    # from plugins.experimental.img2model_era3d import Img2ModelEra3DPlugin
    # from plugins.experimental.txt2model_meshgpt import Txt2ModelMeshGPTPlugin
    # from plugins.experimental.vid2txt_videomae import Vid2TxtVideoMAEPlugin
    # import plugins.experimental.img2img_loopback
    # import plugins.extras.twitter

    quiet = False

    register_plugin(DepthMidasPlugin, quiet)
    register_plugin(DepthAnythingPlugin, quiet)
    register_plugin(DetectYOLOSPlugin, quiet)
    register_plugin(DetectOwlPlugin, quiet)    
    register_plugin(StableDiffusionPlugin, quiet)
    register_plugin(Txt2ImgCannyPlugin, quiet)
    register_plugin(Txt2ImgDepthMidasPlugin, quiet)
    # register_plugin(Txt2ImgOpenPosePlugin, quiet)
    # register_plugin(Txt2ImgInstantIDPlugin, quiet)
    register_plugin(Txt2ImgCascadePlugin, quiet)
    register_plugin(Txt2ImgControlNetPlugin, quiet)
    # register_plugin(Txt2ImgPano360Plugin, quiet)
    # register_plugin(Txt2ImgPhotoMakerPlugin, quiet)
    # register_plugin(Txt2ImgPuLIDPlugin, quiet)
    register_plugin(Txt2ImgRelightPlugin, quiet)
    register_plugin(Txt2ImgZoomPlugin, quiet)
    register_plugin(Txt2ImgFluxPlugin, quiet)
    # register_plugin(Txt2VidVADERPlugin, quiet)
    register_plugin(Txt2VidAnimatePlugin, quiet)
    # register_plugin(Txt2VidAnimateDiffPlugin, quiet)
    register_plugin(Txt2VidZeroscopePlugin, quiet)
    register_plugin(Txt2VidZeroPlugin, quiet)
    register_plugin(Txt2WavStableAudioPlugin, quiet)
    register_plugin(Vid2DensePosePlugin, quiet)
    register_plugin(Vid2VidPlugin, quiet)
    register_plugin(Vid2VidLatentSyncPlugin, quiet)
    # register_plugin(Vid2TxtVideoMAEPlugin, quiet)
    register_plugin(Img2VidGenXLPlugin, quiet)
    register_plugin(Img2VidXTPlugin, quiet)
    register_plugin(Img2VidAniPortraitPlugin, quiet)
    # register_plugin(Txt2VidCogVideoXPlugin, quiet)
    register_plugin(Img2VidLivePortraitPlugin, quiet)
    register_plugin(Vid2VidMagicAnimatePlugin, quiet)
    # register_plugin(VideoAIPlugin, quiet)
    register_plugin(Img2TxtLlavaPlugin, quiet)
    register_plugin(Img2TxtMoondreamPlugin, quiet)
    register_plugin(RembgPlugin, quiet)
    register_plugin(ImgUpresPlugin, quiet)
    register_plugin(Txt2WavMusicGenPlugin, quiet)
    register_plugin(ExllamaV2Plugin, quiet)
    register_plugin(MMAudioPlugin, quiet)
    register_plugin(CausalLMPlugin, quiet)
    register_plugin(Txt2ModelShapEPlugin, quiet)
    register_plugin(Txt2ModelAvatarPlugin, quiet)
    # register_plugin(Txt2ModelMeshGPTPlugin, quiet)
    register_plugin(Img2ModelLGMPlugin, quiet)
    register_plugin(Img2ModelTSRPlugin, quiet)
    # register_plugin(Img2ModelVFusionPlugin, quiet)
    # register_plugin(Img2ModelStableFast3DPlugin, quiet)
    # register_plugin(Img2ModelEra3DPlugin, quiet)
    register_plugin(Img2ModelLGMPlugin, quiet)
    register_plugin(Img2ModelTSRPlugin, quiet)
    register_plugin(TTSPlugin, quiet)
    register_plugin(TxtSummaryPlugin, quiet)
    register_plugin(VoiceWhisperPlugin, quiet)
    register_plugin(VoiceConversationPlugin, quiet)


_lock = Lock()
_start_time: int = None
_plugins: list[Type] = []

router = APIRouter()


class PluginBase:
    name: str = "PluginBase"
    description: str = "Base class for plugins"
    router = router
    instance = None
    plugins = []

    def __init__(
        self,
        deprecated1=None,
        deprecated2=None,
    ):
        if self.__class__.instance is not None:
            raise ValueError("Plugin already instantiated!")

        self.__class__.instance = self
        self.device = autodetect_device()
        self.dtype = autodetect_dtype()
        self.resources: dict[str, object] = {}

    def __str__(self):
        return f"{self.name}: {self.description}"


def register_plugin(plugin_type, quiet=False):
    if plugin_type == PluginBase:
        raise ValueError("Can not register abstract PluginBase as a plugin type")
    if not issubclass(plugin_type, PluginBase):
        raise ValueError(f"Invalid plugin type: {plugin_type}")
    # if plugin_type in _plugins:
    #    raise ValueError(f"Plugin already registered: {plugin_type.name}")

    # _plugins.append(plugin_type)

    _plugins.append(plugin_type)

    if not quiet:
        logging.info(f"Loading plugin: {plugin_type.name}")

    if hasattr(plugin_type, "add_interface"):
        plugin_type.add_interface()


async def use_plugin(plugin_type: type[PluginBase], unsafe: bool = False):
    # see if plugin is in _plugins
    matching_plugin = None if plugin_type not in _plugins else plugin_type

    if matching_plugin is None:
        raise ValueError(f"Invalid plugin type: {plugin_type}")

    global _lock
    global _start_time

    _start_time = time.time()

    gpu_cleared = False

    if unsafe is False:
        await _lock.acquire()

        from plugins.txt2img_flux import Txt2ImgFluxPlugin

        unloaded = False
        for p in _plugins:
            if (
                p != matching_plugin
                and not (p == Txt2ImgFluxPlugin and KEEP_FLUX_LOADED)
                and p.instance is not None
                and p.__name__ not in matching_plugin.plugins
            ):
                unload_plugin(p)
                unloaded = True

        if unloaded:
            clear_gpu_cache()
            gpu_cleared = True

    if not gpu_cleared:
        check_low_vram()

    if matching_plugin.instance is not None:
        log_recycle(f"Reusing plugin: {matching_plugin.name}")
        return matching_plugin.instance

    log_plugin(matching_plugin.name)
    matching_plugin.instance = matching_plugin()

    return matching_plugin.instance


def use_plugin_unsafe(plugin_type: type[PluginBase], reset_time=False):
    # see if plugin is in _plugins
    matching_plugin = None if plugin_type not in _plugins else plugin_type

    if matching_plugin is None:
        raise ValueError(f"Invalid plugin type: {plugin_type}")

    global _start_time

    if reset_time:
        _start_time = time.time()

    if matching_plugin.instance is not None:
        log_recycle(f"Reusing plugin: {matching_plugin.name}")
        check_low_vram()
        return matching_plugin.instance

    log_plugin(matching_plugin.name)
    matching_plugin.instance = matching_plugin()

    check_low_vram()
    return matching_plugin.instance


def release_plugin(plugin: type[PluginBase]):
    global _lock
    global _start_time

    if _start_time is None:
        raise ValueError("No plugin in use")

    elapsed = time.time() - _start_time
    logging.info(
        f"{Emojis.checkmark} Task completed: {plugin.name} in {elapsed:.2f} seconds"
    )
    gc.collect()

    check_low_vram()

    _lock.release()


def check_low_vram():
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory
        reserved_vram = torch.cuda.memory_reserved(0)
        free_vram = total_vram - reserved_vram

        bytes_to_gib
        logging.info(
            f"VRAM: {bytes_to_gib(total_vram):.2f}GiB, {bytes_to_gib(reserved_vram):.2f} used, {free_vram/total_vram*100:.2f}% free"
        )
        if free_vram < 3 * 1024**3:
            logging.warning(
                f"Low GPU memory detected ({free_vram} bytes free), clearing cache"
            )
            clear_gpu_cache()


def _unload_resources(plugin: type[PluginBase]):
    if plugin.instance is None or len(plugin.instance.resources) == 0:
        return

    unload = []

    has_offload = hasattr(plugin.instance, "offload")

    for name, resource in plugin.instance.resources.items():
        if (
            name
            in [
                "model",
                "pipeline",
                "sd",
                "txt2img",
                "img2img",
                "inpaint",
                "tokenizer",
                "cache",
                "streaming_generator",
            ]
            and has_offload
        ):
            continue

        if hasattr(resource, "maybe_free_model_hooks"):
            resource.maybe_free_model_hooks()

        if hasattr(resource, "unload"):
            resource.unload()

        if hasattr(resource, "_model"):  # FILM interpolation etc
            del resource._model

        unload.append(name)

    if len(unload) > 0:
        logging.info(f"Unloading resources from {plugin.name}: {', '.join(unload)}")

    for name in unload:
        del plugin.instance.resources[name]


def unload_plugin(plugin: type[PluginBase]):
    if plugin not in _plugins:
        logging.warn(f"Unload called on unknown plugin: {plugin}")
        return

    if plugin.instance is None:
        return

    if hasattr(plugin.instance, "offload"):
        logging.info(f"Offloading plugin: {plugin.name}")
        plugin.instance.offload()
        _unload_resources(plugin)
        return
    elif hasattr(plugin.instance, "unload"):
        logging.info(f"Unloading plugin: {plugin.name}")
        plugin.instance.unload()
        _unload_resources(plugin)
    else:
        logging.info(f"Purging plugin: {plugin.name}")

    del plugin.instance
    plugin.instance = None
    gc.collect()


def unload_all():
    for plugin in _plugins:
        unload_plugin(plugin)

    clear_gpu_cache()
