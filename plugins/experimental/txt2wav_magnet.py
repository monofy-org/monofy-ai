from fastapi.responses import StreamingResponse
from modules.plugins import PluginBase, use_plugin


class Txt2WavMagnetPlugin(PluginBase):

    name = "txt2wav_magnet"
    description = "Text to Wav using MagnetTTS"
    post_routes = {
        "/txt2wav_magnet": "txt2wav_magnet",
    }
    instance = None

    def __init__(self):

        from submodules.audiocraft.audiocraft.models import MAGNeT

        super().__init__()

        self.resources["model"] = MAGNeT.get_pretrained("facebook/audio-magnet-medium")

    async def generate(
        self,
        prompt: str,
    ):

        from submodules.audiocraft.audiocraft.models import MAGNeT

        plugin = await use_plugin(Txt2WavMagnetPlugin, True)
        model: MAGNeT = plugin.resources["model"]
        wav = model.generate([prompt])
        for idx, one_wav in enumerate(wav):
            yield one_wav.cpu(), model.sample_rate

    async def txt2wav_magnet(
        prompt: str,
    ):
        plugin: Txt2WavMagnetPlugin = await use_plugin(Txt2WavMagnetPlugin, True)
        return StreamingResponse(plugin.generate(prompt), media_type="audio/wav")
