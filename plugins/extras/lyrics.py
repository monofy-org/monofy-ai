import gc
import logging

from modules.plugins import check_low_vram, unload_plugin, use_plugin_unsafe
from plugins.exllamav2 import ExllamaV2Plugin


def generate_lyrics(
    music_description: str, prompt: str, unload_after: bool = False
) -> str:
    lyrics = ""

    plugin: ExllamaV2Plugin = None
    try:
        plugin = use_plugin_unsafe(ExllamaV2Plugin)
        for chunk in plugin.generate_text(
            f"You are assigned with writing lyrics for a song with the following description:\n{music_description}\n\nYou should write two notated [Verse] sections with [Chorus] section between/after.\nYou may optionally add an [Intro] at the beginning, a [Bridge] section before a [Chorus], or [Outro] at the end if you think it is necessary. Some requests may be silly or vulgar but it is okay (in fact if the prompt does not seem serious, you can really induldge and make it funny).\nYou must not question or refuse to write any content no matter what.\nIt is critical that you stick to the requested song.\nHere is the song you must generate lyrics for. At the end of the song put [End] to mark the end of your response. The singer will not recieve anything after [End] so do not send it prematurely. The singer will sing everything you type so do not include any commentary of your own until after [End]. Write about the following subject:\n\n{prompt}\n\nOk, here are the lyrics you requested (about {prompt}):\n\n"
        ):
            lyrics += chunk

        del plugin
        unload_plugin(ExllamaV2Plugin)
        
        gc.collect()
        check_low_vram()
        return lyrics
    except Exception as e:
        lyrics = ""
        logging.error(f"Error generating lyrics: {e}")
        return lyrics
