import logging
import os
import shlex
import subprocess
import tempfile
from typing import Any, Literal, Optional
from pathlib import Path
from fastapi.responses import FileResponse
from huggingface_hub import snapshot_download
from modules.plugins import PluginBase, use_plugin, release_plugin


class ServingConfig:
    huggingface_repo_id: str = "metavoiceio/metavoice-1B-v0.1"
    """Absolute path to the model directory."""

    max_new_tokens: int = 864 * 2
    """Maximum number of new tokens to generate from the first stage model."""

    temperature: float = 1.0
    """Temperature for sampling applied to both models."""

    top_k: int = 200
    """Top k for sampling applied to both models."""

    seed: int = 1337
    """Random seed for sampling."""

    dtype: Literal["bfloat16", "float16", "float32", "tfloat32"] = "bfloat16"
    """Data type to use for sampling."""

    enhancer: Optional[Literal["df"]] = "df"
    """Enhancer to use for post-processing."""

    compile: bool = False
    """Whether to compile the model using PyTorch 2.0."""


class TTSMetavoicePlugin(PluginBase):
    import torch

    name = "tts_metavoice"
    description = "Text-to-speech using MetaVoice"
    post_routes = {
        "/tts/metavoice": "text_to_speech",
    }

    def __init__(self):
        import torch
        from submodules.metavoice.fam.llm.sample import (
            InferenceConfig,
            build_models,
            get_first_stage_path,
            get_second_stage_path,
        )

        common_config = dict(
            num_samples=1,
            seed=1337,
            device=TTSMetavoicePlugin.device,
            dtype="bfloat16",
            compile=False,
            init_from="resume",
            output_dir=tempfile.mkdtemp(),
        )

        model_dir = snapshot_download(repo_id=ServingConfig.huggingface_repo_id)

        config1 = InferenceConfig(
            ckpt_path=get_first_stage_path(model_dir),
            **common_config,
        )

        config2 = InferenceConfig(
            ckpt_path=get_second_stage_path(model_dir),
            **common_config,
        )

        if common_config["compile"] is True:
            first_stage_model = torch.jit.load(get_first_stage_path())
            second_stage_model = torch.jit.load(get_second_stage_path())
        else:
            spkemb_model, first_stage_model, second_stage_model = build_models(
                config_first_stage=config1,
                config_second_stage=config2,
                model_dir=model_dir,
                device=TTSMetavoicePlugin.device,
                use_kv_cache="flash_decoding",  # "vanilla", "flash_decoding", or None
            )

        print(f"spkemb_model: {spkemb_model}")

        # if config.enhancer is not None:
        #    from submodules.metavoice.fam.enhance import build_enhancer
        #
        #    enhancer = build_enhancer(config.enhancer)
        # else:
        enhancer = None

        resources = {
            "config1": config1,
            "config2": config2,
            "first_stage_model": first_stage_model,
            "second_stage_model": second_stage_model,
            "spkemb_model": spkemb_model,
        }
        if enhancer is not None:
            resources["enhancer"] = enhancer

        super().__init__(None, resources)

    async def text_to_speech(text: str, voice: str = "female1"):

        plugin: TTSMetavoicePlugin = await use_plugin(TTSMetavoicePlugin)

        wav_out_path = None

        try:
            spkemb_model = plugin.resources["spkemb_model"]
            first_stage_model = plugin.resources["first_stage_model"]
            second_stage_model = plugin.resources["second_stage_model"]
            enhancer = plugin.resources.get("enhancer", None)

            wav_out_path = await plugin.generate_speech(
                text,
                spkemb_model,
                first_stage_model,
                second_stage_model,
                enhancer,
                ServingConfig.temperature,
                ServingConfig.max_new_tokens,
                voice,
            )
            if os.path.exists(wav_out_path):
                return FileResponse(wav_out_path, media_type="audio/wav")
            else:
                raise ValueError("Failed to generate audio")
        except Exception as e:
            logging.error(e, exc_info=True)
            raise e
        finally:
            release_plugin(TTSMetavoicePlugin)
            if wav_out_path is not None:
                Path(wav_out_path).unlink(missing_ok=True)

    async def generate_speech(
        self,
        text: str,
        spkemb_model: torch.nn.Module,
        first_stage_model: Any,
        second_stage_model: Any,
        enhancer: object,
        temperature: float,
        max_new_tokens: int,
        voice: str,
        top_k: int = None,
        top_p: float = None,
        audiodata: bytes = None,
        guidance: bytes = None,
    ):
        from submodules.metavoice.fam.llm.sample import sample_utterance

        wav_out_path = None

        try:

            with tempfile.NamedTemporaryFile(suffix=".wav") as wav_tmp:
                if voice is None:
                    wav_path = _convert_audiodata_to_wav_path(audiodata, wav_tmp)
                else:
                    wav_path = os.path.join("voices", f"{voice}.wav")
                if not os.path.exists(wav_path):
                    raise ValueError(f"Voice {voice} not found")
                if wav_path is None:
                    logging.warn("Running without speaker reference")
                    assert guidance is None
                wav_out_path = sample_utterance(
                    text,
                    wav_path,
                    spkemb_model,
                    first_stage_model,
                    second_stage_model,
                    enhancer,
                    None,
                    None,
                    guidance,
                    temperature,
                    max_new_tokens,
                    None,
                    0.9,
                )

                return wav_out_path
        except Exception as e:
            # traceback_str = "".join(traceback.format_tb(e.__traceback__))
            logging.error(e, exc_info=True)
            raise e


# Singleton
class _GlobalState:
    import torch

    spkemb_model: torch.nn.Module
    first_stage_model: Any
    second_stage_model: Any
    config: ServingConfig
    enhancer: object


GlobalState = _GlobalState()


def _convert_audiodata_to_wav_path(audiodata, wav_tmp):
    with tempfile.NamedTemporaryFile() as unknown_format_tmp:
        if unknown_format_tmp.write(audiodata) == 0:
            return None
        unknown_format_tmp.flush()

        subprocess.check_output(
            # arbitrary 2 minute cutoff
            shlex.split(
                f"ffmpeg -t 120 -y -i {unknown_format_tmp.name} -f wav {wav_tmp.name}"
            )
        )

        return wav_tmp.name
