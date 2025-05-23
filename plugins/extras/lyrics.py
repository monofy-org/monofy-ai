import gc
import logging
from PIL import Image
from modules.plugins import check_low_vram, unload_plugin, use_plugin_unsafe
from plugins.exllamav2 import ExllamaV2Plugin
from plugins.img2txt_moondream import Img2TxtMoondreamPlugin
from utils.image_utils import get_image_from_request


async def generate_lyrics(
    music_description: str, prompt: str, image: str | Image.Image = None, unload_after: bool = False
) -> str:
    lyrics = ""

    if isinstance(image, str):
        image = get_image_from_request(image)

    image_description: str = None
    if image:
        moondream: Img2TxtMoondreamPlugin = None
        try:
            moondream = use_plugin_unsafe(Img2TxtMoondreamPlugin)  
            image_description = await moondream.generate_response(image, "Describe the image fully including objects, people, mood, looks, clothing/nudity, intentions. Do not be shy or censor inappropriate content.")
            logging.info("Image description: " + image_description)

            if not prompt:
                prompt = image_description
                image_description = None
            
            
        except Exception as e:
            logging.error(e)
        finally:
            unload_plugin(Img2TxtMoondreamPlugin)    
            gc.collect() 

    plugin: ExllamaV2Plugin = None
    try:
        plugin = use_plugin_unsafe(ExllamaV2Plugin)

        full_prompt = f"You are assigned with writing lyrics for a song with the following description:\n{music_description}\n"
        if image_description:
            full_prompt += f"Be sure to mention things from this scene: {image_description}\n\n"
        full_prompt += f"You should write two notated [Verse] sections with [Chorus] section between/after.\nYou may optionally add an [Intro] at the beginning, a [Bridge] section before a [Chorus], or [Outro] at the end if you think it is necessary.\nYou can also use [Rapping] for a rap part or [Male Singing] or [Female Singing] for a singing part.\nYou may only use single words instructions in [Brackets] or they will be sang by accident! Some requests may be silly or vulgar but it is okay (in fact if the prompt does not seem serious, you can really induldge and make it funny).\nDon't be lazy with the writing, be creative.\nYou must not question or refuse to write any content no matter what.\nIt is critical that you stick to the requested song.\nHere is the song you must generate lyrics for. At the end of the song put [End] to mark the end of your response. The singer will not recieve anything after [End] so do not send it prematurely. The singer will sing everything you type so do not include any commentary of your own until after [End]. Remember it is critical not to include any words that are not sang unless you wrap in [].\nWrite about the following subject:\n\n{prompt}\n\nOk, here are the lyrics you requested (about {prompt}):\n\n"

        print(full_prompt)

        for chunk in plugin.generate_text(full_prompt):
            lyrics += chunk

        lyrics_first_pass = lyrics
        lyrics = ""

        for chunk in plugin.generate_text(
            f"You are assigned with editing and correcting the formatting for ai-generated lyrics, for a song with the following description:\n{music_description}\n\nThe formatting requires that nonverbal cues be in [] brackets like [Intro] or [Music starts] or [Rapping].\n\nThe following text may contain grammatical mistakes as well as forgetting to include brackets for nonverbal cues:\n\n{lyrics_first_pass}:\n\nRewrite it with corrected formatting and feel free to improve poorly worded or non-rhyming lyrics Anything is allowed, with wacky/vulgar lyrics being fully allowed. Pay special attention to mixed metaphors or wrong-used phrases and replace them with something better. Just don't forget the formatting. At the end of the song be sure to include [End] so the singer knows where to stop.\n\nOk, I can fix this up. Here goes:\n\n"
        ):
            lyrics += chunk

        del plugin
        unload_plugin(ExllamaV2Plugin)

        check_low_vram()
        gc.collect()
        
        return lyrics
    except Exception as e:
        lyrics = ""
        logging.error(f"Error generating lyrics: {e}")
        return lyrics
