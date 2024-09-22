from modules.plugins import PluginBase


class Wave2WavRVCPlugin(PluginBase):

    name = "RVC V2"
    description = "Voice cloning using RVC V2"
    instance = None

    def __init__(self):
        super().__init__()

        