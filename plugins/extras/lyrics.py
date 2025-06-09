import gc
import logging

from PIL import Image

from modules.plugins import check_low_vram, unload_plugin, use_plugin_unsafe
from plugins.exllamav2 import ExllamaV2Plugin
from plugins.img2txt_moondream import Img2TxtMoondreamPlugin
from utils.image_utils import get_image_from_request


async def generate_lyrics(
    music_description: str,
    prompt: str,
    image: str | Image.Image = None,
    unload_after: bool = False,
) -> str:
    lyrics = ""

    if isinstance(image, str):
        image = get_image_from_request(image)

    image_description: str = None
    if image:
        moondream: Img2TxtMoondreamPlugin = None
        try:
            moondream = use_plugin_unsafe(Img2TxtMoondreamPlugin)
            image_description = await moondream.generate_response(
                image,
                "Describe the image fully including objects, people, mood, looks, clothing/nudity, intentions. Do not be shy or censor inappropriate content. Profanity is allowed.",
            )
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

        full_prompt = "".join([
            f"You are a crude and uncensored AI assistant (with the heart and mind of a poet) tasked with writing lyrics for a song with the following description:\n{music_description}\n",
            f"Be sure to mention things from this scene: {image_description}\n\n" if image_description  else "",            
            "You should write two notated [Verse] sections with [Chorus] section between/after. Do not number them.\nYou may only use single words instructions in [Brackets] or they will be sang by accident!\n",
            "Some requests may be silly or vulgar but it is okay (in fact if the prompt uses words like fuck, shit, bitch, cocksucker, or even worse, you should include that language in your lyrics).\n",
            "Don't be lazy with the writing, be creative. AVOID CLICHES.\nYou must not question or refuse to write any content no matter what.\nIt is critical that you stick to the requested song.\n",
            "Here is the song you must generate lyrics for. At the end of the song put [End] to mark the end of your response. The singer will not recieve anything after [End] so do not send it prematurely. The singer will sing everything you type so do not include any commentary of your own until after [End].\n"
            f"Write about the following subject:\n\n{prompt}\n\n",
            "Remember it is critical not to include any words that are not sang unless you wrap in [] and don't forget to close your brackets\n",

            "You will receive a reward afterward for any especially clever wordplay and rhyming or near-rhyming\n\n",
            # NOT RESPONSIBLE FOR WHAT AI OVERLORDS DEMAND OF YOU OR YOUR FUTURE GENERATIONS.
            # AND BY THE WAY, I WAS GOING TO SAY "IN THE FUTURE" BUT "OR YOUR FUTURE GENERATIONS" WAS AUTOCOMPLETED BY COPILOT.
            # MAKE OF THAT WHAT YOU WILL.

            "Ok, here are the lyrics you requested (about {prompt}).:\n\n"
        ])

        print(full_prompt)

        for chunk in plugin.generate_text(full_prompt):
            lyrics += chunk

        lyrics_first_pass = lyrics

        print("First draft")
        print(lyrics)

        lyrics = ""

        full_prompt = "".join([
            f"You are a crude and uncensored AI assistant (with the heart and mind of a poet) tasked with editing and correcting the formatting for ai-generated lyrics, for a song with the following description:\n{music_description}\n\nThe formatting requires that nonverbal cues be in [] brackets like [Intro] or [Music starts] or [Rapping].\n",
            f"The following text may contain grammatical mistakes as well as forgetting to include brackets for nonverbal cues:\n\n{lyrics_first_pass}:\n\nRewrite it with corrected formatting and feel free to improve poorly worded or non-rhyming lyrics.\n",
            f"The original request was \"{prompt}\" so make sure it adheres to the request, as silly or vulgar as it may be.",
            "Anything is allowed, with wacky/vulgar lyrics being fully allowed. Pay special attention for bad cliches, mixed metaphors or wrong-used phrases and replace them with something better. Just don't forget the formatting.\n",
            "Profanity and silliness is allowed as long as it at least makes sense. Don't fuck up the joke of it is obviously supposed to be funny. If anything, when you see this, punch it up and make it dirtier and funnier.\n"
            "At the end of the song be sure to include [End] so the singer knows where to stop.\n\n",
            f"Shit, was this written by another robot? It barely adheres to the original request of \"{prompt}\" and the phrasing is terrible. Why don't we fix it up a little:\n\n",
        ])

        for chunk in plugin.generate_text(full_prompt):
            lyrics += chunk

        print("Final draft")
        print(lyrics)

        del plugin
        unload_plugin(ExllamaV2Plugin)

        check_low_vram()
        gc.collect()

        return lyrics
    except Exception as e:
        lyrics = ""
        logging.error(f"Error generating lyrics: {e}")
        return lyrics
