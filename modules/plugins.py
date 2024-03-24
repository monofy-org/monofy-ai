import gc
import logging
import time
from fastapi.routing import APIRoute, APIRouter
from fastapi.utils import generate_unique_id
from asyncio import Lock

import torch
from utils.gpu_utils import autodetect_device, autodetect_dtype, clear_gpu_cache


def load_plugins():
    from plugins.img_canny import CannyPlugin
    from plugins.stable_diffusion import StableDiffusionPlugin
    from plugins.txt2img_canny import Txt2ImgCannyPlugin
    from plugins.txt2img_depth import Txt2ImgDepthMidasPlugin
    from plugins.txt2img_instantid import Txt2ImageInstantIDPlugin
    from plugins.txt2img_cascade import Txt2ImgCascadePlugin
    from plugins.txt2img_controlnet import Txt2ImgControlNetPlugin
    from plugins.img_depth_anything import DepthAnythingPlugin
    from plugins.img_depth_midas import DepthPlugin
    from plugins.detect_yolos import DetectYOLOSPlugin
    from plugins.img2model_lgm import Img2ModelLGMPlugin
    from plugins.img2model_tsr import Img2ModelTSRPlugin
    from plugins.txt2vid import Txt2VidZeroPlugin
    from plugins.img2vid_xt import Img2VidXTPlugin
    from plugins.img_rembg import RembgPlugin
    from plugins.vision import VisionPlugin
    from plugins.musicgen import MusicGenPlugin
    from plugins.exllamav2 import ExllamaV2Plugin
    from plugins.shap_e import Txt2ModelShapEPlugin    
    from plugins.tts import TTSPlugin
    from plugins.txt2vid_animate import Txt2VidAnimatePlugin
    from plugins.txt2vid_zeroscope import Txt2VidZeroscopePlugin
    from plugins.youtube import YouTubePlugin
    from plugins.txt_summary import TxtSummaryPlugin

    register_plugin(CannyPlugin, True)
    register_plugin(DepthPlugin, True)
    register_plugin(DepthAnythingPlugin, True)
    register_plugin(DetectYOLOSPlugin, True)
    register_plugin(StableDiffusionPlugin, True)
    register_plugin(Txt2ImgCannyPlugin, True)
    register_plugin(Txt2ImgDepthMidasPlugin, True)
    register_plugin(Txt2ImageInstantIDPlugin, True)
    register_plugin(Txt2ImgCascadePlugin, True)
    register_plugin(Txt2ImgControlNetPlugin, True)
    register_plugin(Txt2VidZeroPlugin, True)
    register_plugin(Txt2VidAnimatePlugin, True)
    register_plugin(Txt2VidZeroscopePlugin, True)
    register_plugin(Img2VidXTPlugin, True)
    register_plugin(Img2ModelLGMPlugin, True)
    register_plugin(Img2ModelTSRPlugin, True)
    register_plugin(RembgPlugin, True)
    register_plugin(VisionPlugin, True)
    register_plugin(MusicGenPlugin, True)
    register_plugin(ExllamaV2Plugin, True)
    register_plugin(Txt2ModelShapEPlugin, True)
    register_plugin(Img2ModelLGMPlugin, True)
    register_plugin(Img2ModelTSRPlugin, True)
    register_plugin(TTSPlugin, True)
    register_plugin(YouTubePlugin, True)
    register_plugin(TxtSummaryPlugin, True)


_lock = Lock()
_start_time: int = None
_plugins = []

router = APIRouter()


class PluginBase:

    name: str = "PluginBase"
    description: str = "Base class for plugins"
    router = router
    instance = None

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

    #LEGACY        
    if hasattr(plugin_type, "post_routes"):
        logging.warning(f"Plugin {plugin_type.name} is using legacy route definitions")
        for path, function_name in plugin_type.post_routes.items():    
            logging.info(
                f"Adding route (POST): {path} -> {plugin_type.__name__}.{function_name}"
            )

            # endpoints are static plugin class methods
            endpoint = plugin_type.__dict__[function_name]

            post = APIRoute(
                path=path,
                endpoint=endpoint,
                methods=["POST"],
            )
            post.operation_id = generate_unique_id(post)
            PluginBase.router.routes.append(post)

    #LEGACY
            
    if hasattr(plugin_type, "get_routes"):
        logging.warning(f"Plugin {plugin_type.name} is using legacy route definitions")
        for path, function_name in plugin_type.get_routes.items():
            logging.info(
                f"Adding route (GET): {path} -> {plugin_type.__name__}.{function_name}"
            )

            # endpoints are static plugin class methods
            endpoint = plugin_type.__dict__[function_name]

            get = APIRoute(
                path=path,
                endpoint=endpoint,
                methods=["GET"],
            )
            get.operation_id = generate_unique_id(get)
            PluginBase.router.routes.append(get)


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
            if p != matching_plugin and p.instance is not None:
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
