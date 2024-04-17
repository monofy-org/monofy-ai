import gc
import logging
import time
from typing import Type
from fastapi.routing import APIRouter
from asyncio import Lock
from utils.gpu_utils import autodetect_device, autodetect_dtype, clear_gpu_cache


def load_plugins():
    from plugins.stable_diffusion import StableDiffusionPlugin
    from plugins.txt2img_canny import Txt2ImgCannyPlugin
    from plugins.txt2img_depth import Txt2ImgDepthMidasPlugin
    from plugins.txt2img_instantid import Txt2ImgInstantIDPlugin
    from plugins.txt2img_cascade import Txt2ImgCascadePlugin
    from plugins.txt2img_controlnet import Txt2ImgControlNetPlugin
    from plugins.txt2vid_animate import Txt2VidAnimatePlugin
    from plugins.txt2vid_zeroscope import Txt2VidZeroscopePlugin
    from plugins.img2vid_xt import Img2VidXTPlugin
    from plugins.txt2vid import Txt2VidZeroPlugin
    from plugins.img_depth_anything import DepthAnythingPlugin
    from plugins.img_depth_midas import DepthMidasPlugin
    from plugins.detect_yolos import DetectYOLOSPlugin
    from plugins.img2model_lgm import Img2ModelLGMPlugin
    from plugins.img2model_tsr import Img2ModelTSRPlugin
    from plugins.img_rembg import RembgPlugin
    from plugins.img2txt_moondream import Img2TxtMoondreamPlugin
    from plugins.img2txt_llava import Img2TxtLlavaPlugin
    from plugins.musicgen import MusicGenPlugin
    from plugins.exllamav2 import ExllamaV2Plugin
    from plugins.txt2model_shap_e import Txt2ModelShapEPlugin
    from plugins.txt2model_avatar import Txt2ModelAvatarPlugin
    from plugins.tts import TTSPlugin    
    from plugins.txt_summary import TxtSummaryPlugin
    from plugins.voice_whisper import VoiceWhisperPlugin
    from plugins.voice_conversation import VoiceConversationPlugin
    from plugins.experimental.vid2vid_frames import Vid2VidPlugin
    import plugins.experimental.img2img_loopback
    import plugins.extras.txt_profile
    import plugins.extras.txt2img_face
    import plugins.extras.img_canny
    import plugins.extras.img_exif
    import plugins.extras.pdf_rip
    import plugins.extras.youtube
    import plugins.extras.google_trends

    quiet = False

    register_plugin(DepthMidasPlugin, quiet)
    register_plugin(DepthAnythingPlugin, quiet)
    register_plugin(DetectYOLOSPlugin, quiet)
    register_plugin(StableDiffusionPlugin, quiet)
    register_plugin(Txt2ImgCannyPlugin, quiet)
    register_plugin(Txt2ImgDepthMidasPlugin, quiet)
    register_plugin(Txt2ImgInstantIDPlugin, quiet)
    register_plugin(Txt2ImgCascadePlugin, quiet)
    register_plugin(Txt2ImgControlNetPlugin, quiet)
    register_plugin(Txt2VidZeroPlugin, quiet)
    register_plugin(Txt2VidAnimatePlugin, quiet)
    register_plugin(Txt2VidZeroscopePlugin, quiet)
    register_plugin(Vid2VidPlugin, quiet)
    register_plugin(Img2VidXTPlugin, quiet)
    register_plugin(Img2ModelLGMPlugin, quiet)
    register_plugin(Img2ModelTSRPlugin, quiet)
    register_plugin(Img2TxtLlavaPlugin, quiet)
    register_plugin(Img2TxtMoondreamPlugin, quiet)
    register_plugin(RembgPlugin, quiet)
    register_plugin(MusicGenPlugin, quiet)
    register_plugin(ExllamaV2Plugin, quiet)
    register_plugin(Txt2ModelShapEPlugin, quiet)
    register_plugin(Txt2ModelAvatarPlugin, quiet)
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

    if unsafe is False:
        await _lock.acquire()

        unloaded = False
        for p in _plugins:
            if (
                p != matching_plugin
                and p.instance is not None
                and p.__name__ not in matching_plugin.plugins
            ):
                unload_plugin(p)
                unloaded = True

        if unloaded:
            clear_gpu_cache()

    if matching_plugin.instance is not None:
        logging.info(f"Reusing plugin: {matching_plugin.name}")
        return matching_plugin.instance

    logging.info(f"Using plugin: {matching_plugin.name}")
    matching_plugin.instance = matching_plugin()

    return matching_plugin.instance


def release_plugin(plugin: type[PluginBase]):
    import torch

    global _lock
    global _start_time

    if _start_time is None:
        raise ValueError("No plugin in use")

    elapsed = time.time() - _start_time
    logging.info(f"Task completed: {plugin.name} in {elapsed:.2f} seconds")
    gc.collect()

    # if free vram is less than 3gb, clear cache
    if torch.cuda.is_available():
        free_vram = torch.cuda.memory_reserved()
        if free_vram < 3 * 1024**3:
            clear_gpu_cache()

    _lock.release()


def _unload_resources(plugin: type[PluginBase]):

    if plugin.instance is None or len(plugin.instance.resources) == 0:
        return

    unload = []

    for name, resource in plugin.instance.resources.items():
        if hasattr(resource, "unload"):
            resource.unload()
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

    logging.info(f"Offloading plugin: {plugin.name}")

    _unload_resources(plugin)

    for item in list(plugin.instance.resources.values()):

        if hasattr(item, "maybe_free_model_hooks"):
            item.maybe_free_model_hooks()

        if hasattr(item, "unload"):
            item.unload()

        if hasattr(item, "model"):
            del item.model

        if hasattr(item, "tokenizer"):
            del item.tokenizer

        if hasattr(item, "text_encoder"):
            del item.text_encoder

        if hasattr(item, "unet"):
            del item.unet

        if hasattr(item, "vae"):
            del item.vae

        del item

    del plugin.instance
    plugin.instance = None
