import torch
import gguf
from transformers import AutoTokenizer
from PIL import Image
from .ops import GGMLTensor, GGMLOps, move_patch_to_device
from .dequant import is_quantized, dequantize_tensor

class FluxGGUFPipeline:
    def __init__(self, unet_path, clip_path, vae_path=None, scheduler=None, device="cuda"):
        self.device = device
        self.ops = GGMLOps()
        
        # Load models
        self.unet = self._load_gguf_unet(unet_path)
        self.text_encoder = self._load_gguf_clip(clip_path)
        self.vae = self._load_gguf_vae(vae_path) if vae_path else None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(clip_path)
        
        # Set up scheduler
        self.scheduler = scheduler or self._default_scheduler()
        
    def _load_gguf_unet(self, unet_path):
        sd = gguf_sd_loader(unet_path)
        model = load_diffusion_model_state_dict(sd, model_options={"custom_operations": self.ops})
        return GGUFModelPatcher(model, self.device)
    
    def _load_gguf_clip(self, clip_path):
        clip_data = gguf_clip_loader(clip_path)
        clip = load_text_encoder_state_dicts(
            clip_type=6, # CLIPType.Flux
            state_dicts=[clip_data],
            model_options={"custom_operations": self.ops}
        )
        return GGUFModelPatcher(clip.patcher, self.device)
    
    def _load_gguf_vae(self, vae_path):
        sd = gguf_sd_loader(vae_path)
        vae = load_vae_state_dict(sd, model_options={"custom_operations": self.ops})
        return GGUFModelPatcher(vae, self.device)
    
    def _default_scheduler(self):
        from diffusers import FluxScheduler
        return FluxScheduler()
    
    @torch.no_grad()
    def __call__(self, prompt, negative_prompt="", num_inference_steps=50, guidance_scale=7.5, width=512, height=512, generator=None):
        # Encode text
        text_embeddings = self._encode_prompt(prompt, negative_prompt)
        
        # Prepare latents
        latents = self._prepare_latents(width, height, generator)
        
        # Denoising loop
        for i, t in enumerate(self.scheduler.timesteps):
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)
            
            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute previous noisy sample
            latents = self.scheduler.step(noise_pred, t, latents, generator=generator).prev_sample
        
        # Decode latents
        image = self._decode_latents(latents)
        
        return self._postprocess_image(image)
    
    def _encode_prompt(self, prompt, negative_prompt):
        # Tokenize prompts
        text_input = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        uncond_input = self.tokenizer(negative_prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        
        # Encode prompts
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        
        # Concatenate for classifier-free guidance
        return torch.cat([uncond_embeddings, text_embeddings])
    
    def _prepare_latents(self, width, height, generator):
        latents_shape = (1, self.unet.in_channels, height // 8, width // 8)
        latents = torch.randn(latents_shape, generator=generator, device=self.device)
        return latents * self.scheduler.init_noise_sigma
    
    def _decode_latents(self, latents):
        if self.vae:
            return self.vae.decode(latents / 0.18215).sample
        else:
            # If no VAE, return raw latents (this should be improved based on Flux specifics)
            return latents
    
    def _postprocess_image(self, image):
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        return Image.fromarray((image[0] * 255).round().astype("uint8"))

class GGUFModelPatcher:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.patches = {}
    
    def __call__(self, *args, **kwargs):
        # Move inputs to the correct device
        args = [arg.to(self.device) if isinstance(arg, torch.Tensor) else arg for arg in args]
        kwargs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        
        # Apply patches (if any)
        self._apply_patches()
        
        # Forward pass
        output = self.model(*args, **kwargs)
        
        # Undo patches
        self._undo_patches()
        
        return output
    
    def _apply_patches(self):
        for name, patch in self.patches.items():
            original = getattr(self.model, name)
            setattr(self.model, name, patch)
            self.patches[name] = (patch, original)
    
    def _undo_patches(self):
        for name, (patch, original) in self.patches.items():
            setattr(self.model, name, original)
            self.patches[name] = patch

def gguf_sd_loader(path, handle_prefix="model.diffusion_model."):
    reader = gguf.GGUFReader(path)
    state_dict = {}
    for tensor in reader.tensors:
        if handle_prefix and tensor.name.startswith(handle_prefix):
            sd_key = tensor.name[len(handle_prefix):]
        else:
            sd_key = tensor.name
        torch_tensor = torch.from_numpy(tensor.data)
        state_dict[sd_key] = GGMLTensor(torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=torch.Size(tensor.shape))
    return state_dict

def gguf_clip_loader(path):
    return gguf_sd_loader(path, handle_prefix=None)

def load_vae_state_dict(state_dict, model_options=None):
    # This function should be implemented based on the specific VAE architecture used in Flux
    # For now, it's a placeholder
    raise NotImplementedError("VAE loading is not implemented yet.")