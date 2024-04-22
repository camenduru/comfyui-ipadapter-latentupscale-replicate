import os, sys
from cog import BasePredictor, Input, Path
from typing import List
sys.path.append('/content/TotoroUI')
sys.path.append('/content/TotoroUI/IPAdapter')
os.chdir('/content/TotoroUI')

import torch
import numpy as np
from PIL import Image
import totoro
import scipy
from latent_resizer import LatentResizer
import gc

def upscale(latent, upscale, model, device, vae_device):
  samples = latent.to(device=device, dtype=torch.float16)
  model.to(device=device)
  latent_out = (model(0.13025 * samples, scale=upscale) / 0.13025)
  latent_out = latent_out.to(device="cpu")
  model.to(device=vae_device)
  return ({"samples": latent_out},)

# mask_from_colors() and conditioning_combine_multiple() from https://github.com/cubiq/ComfyUI_essentials/blob/main/essentials.py
def mask_from_colors(image, threshold_r, threshold_g, threshold_b, remove_isolated_pixels, fill_holes):
    red = ((image[..., 0] >= 1-threshold_r) & (image[..., 1] < threshold_g) & (image[..., 2] < threshold_b)).float()
    green = ((image[..., 0] < threshold_r) & (image[..., 1] >= 1-threshold_g) & (image[..., 2] < threshold_b)).float()
    blue = ((image[..., 0] < threshold_r) & (image[..., 1] < threshold_g) & (image[..., 2] >= 1-threshold_b)).float()
    cyan = ((image[..., 0] < threshold_r) & (image[..., 1] >= 1-threshold_g) & (image[..., 2] >= 1-threshold_b)).float()
    magenta = ((image[..., 0] >= 1-threshold_r) & (image[..., 1] < threshold_g) & (image[..., 2] > 1-threshold_b)).float()
    yellow = ((image[..., 0] >= 1-threshold_r) & (image[..., 1] >= 1-threshold_g) & (image[..., 2] < threshold_b)).float()
    black = ((image[..., 0] <= threshold_r) & (image[..., 1] <= threshold_g) & (image[..., 2] <= threshold_b)).float()
    white = ((image[..., 0] >= 1-threshold_r) & (image[..., 1] >= 1-threshold_g) & (image[..., 2] >= 1-threshold_b)).float()
    if remove_isolated_pixels > 0 or fill_holes:
        colors = [red, green, blue, cyan, magenta, yellow, black, white]
        color_names = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'white']
        processed_colors = {}
        for color_name, color in zip(color_names, colors):
            color = color.cpu().numpy()
            masks = []

            for i in range(image.shape[0]):
                mask = color[i]
                if remove_isolated_pixels > 0:
                    mask = scipy.ndimage.binary_opening(mask, structure=np.ones((remove_isolated_pixels, remove_isolated_pixels)))
                if fill_holes:
                    mask = scipy.ndimage.binary_fill_holes(mask)
                mask = torch.from_numpy(mask)
                masks.append(mask)
            processed_colors[color_name] = torch.stack(masks, dim=0).float()
        red = processed_colors['red']
        green = processed_colors['green']
        blue = processed_colors['blue']
        cyan = processed_colors['cyan']
        magenta = processed_colors['magenta']
        yellow = processed_colors['yellow']
        black = processed_colors['black']
        white = processed_colors['white']
        del colors, processed_colors
    return (red, green, blue, cyan, magenta, yellow, black, white,)

def conditioning_combine_multiple(conditioning_1, conditioning_2, conditioning_3=None, conditioning_4=None, conditioning_5=None):
    c = conditioning_1 + conditioning_2
    if conditioning_3 is not None:
        c += conditioning_3
    if conditioning_4 is not None:
        c += conditioning_4
    if conditioning_5 is not None:
        c += conditioning_5
    return (c,)

class Predictor(BasePredictor):
    def setup(self) -> None:
        import IPAdapterPlus
        from totoro import model_management
        with torch.no_grad():
            self.device = model_management.get_torch_device()
            self.vae_device = model_management.vae_offload_device()
            self.model_up = LatentResizer.load_model('/content/TotoroUI/models/sd15_resizer.pt', self.device, torch.float16)
            self.model_patcher, self.clip, self.vae, self.clipvision = totoro.sd.load_checkpoint_guess_config("/content/TotoroUI/models/dreamshaper_8.safetensors", output_vae=True, output_clip=True, embedding_directory=None)
            self.IPAdapterPlus_model = IPAdapterPlus.IPAdapterUnifiedLoader().load_models(self.model_patcher, 'PLUS (high strength)', lora_strength=0.0, provider="CPU", ipadapter=None)
    def predict(
        self,
        green_part: Path = Input(description="Green part"),
        red_part: Path = Input(description="Red part"),
        black_part: Path = Input(description="Black part"),
        color_mask: Path = Input(description="Color Mask"),
        latent_upscale: bool = Input(False, description="Latent Upscale"),
    ) -> List[Path]:
        import nodes, IPAdapterPlus
        from totoro import model_management
        with torch.no_grad():
            output1_image, output1_mask = nodes.LoadImage().load_image(str(red_part))
            output2_image, output2_mask = nodes.LoadImage().load_image(str(green_part))
            output3_image, output3_mask = nodes.LoadImage().load_image(str(black_part))
            color_image, color_mask = nodes.LoadImage().load_image(str(color_mask))
            red, green, blue, cyan, magenta, yellow, black, white = mask_from_colors(image=color_image, threshold_r=0.15, threshold_g=0.15, threshold_b=0.15, remove_isolated_pixels=0, fill_holes=False)

            tokens_1 = self.clip.tokenize("illustration of a blond woman")
            cond_1, pooled_1 = self.clip.encode_from_tokens(tokens_1, return_pooled=True)
            cond_1 = [[cond_1, {"pooled_output": pooled_1}]]
            n_tokens_1 = self.clip.tokenize("anime")
            n_cond_1, n_pooled_1 = self.clip.encode_from_tokens(n_tokens_1, return_pooled=True)
            n_cond_1 = [[n_cond_1, {"pooled_output": n_pooled_1}]]
            params_1, positive_1, negative_1 = IPAdapterPlus.IPAdapterRegionalConditioning().conditioning(output1_image, image_weight=0.7, prompt_weight=1.0, weight_type='linear', start_at=0.0, end_at=1.0, mask=red, positive=cond_1, negative=n_cond_1)

            tokens_2 = self.clip.tokenize("anime illustration of a young woman with a black jacket")
            cond_2, pooled_2 = self.clip.encode_from_tokens(tokens_2, return_pooled=True)
            cond_2 = [[cond_2, {"pooled_output": pooled_2}]]
            n_tokens_2 = self.clip.tokenize("")
            n_cond_2, n_pooled_2 = self.clip.encode_from_tokens(n_tokens_2, return_pooled=True)
            n_cond_2 = [[n_cond_2, {"pooled_output": n_pooled_2}]]
            params_2, positive_2, negative_2 = IPAdapterPlus.IPAdapterRegionalConditioning().conditioning(output2_image, image_weight=0.7, prompt_weight=1.0, weight_type='linear', start_at=0.0, end_at=1.0, mask=green, positive=cond_2, negative=n_cond_2)

            tokens_3 = self.clip.tokenize("closeup of two girl friends shopping in a sci-fi space station")
            cond_3, pooled_3 = self.clip.encode_from_tokens(tokens_3, return_pooled=True)
            cond_3 = [[cond_3, {"pooled_output": pooled_3}]]
            n_tokens_3 = self.clip.tokenize("blurry, lowres, bad art, ill, distorted, malformed, horror")
            n_cond_3, n_pooled_3 = self.clip.encode_from_tokens(n_tokens_3, return_pooled=True)
            n_cond_3 = [[n_cond_3, {"pooled_output": n_pooled_3}]]
            params_3, positive_3, negative_3 = IPAdapterPlus.IPAdapterRegionalConditioning().conditioning(output3_image, image_weight=0.7, prompt_weight=1.0, weight_type='linear', start_at=0.0, end_at=1.0, mask=black, positive=None, negative=None)
            positive = conditioning_combine_multiple(conditioning_1=positive_1, conditioning_2=positive_2, conditioning_3=cond_3)
            negative = conditioning_combine_multiple(conditioning_1=negative_1, conditioning_2=negative_2, conditioning_3=n_cond_3)
            ipadapter_params = IPAdapterPlus.IPAdapterCombineParams().combine(params_1=params_1, params_2=params_2, params_3=params_3)
            ip_model_patcher = IPAdapterPlus.IPAdapterAdvanced().apply_ipadapter(self.IPAdapterPlus_model[0], self.IPAdapterPlus_model[1], start_at=0.0, end_at=1.0, weight=1.0, weight_style=1.0, weight_composition=1.0, expand_style=False, weight_type="linear", combine_embeds="concat", embeds_scaling='V only', ipadapter_params=ipadapter_params[0])
            latent = {"samples":torch.zeros([1, 4, 512 // 8, 768 // 8])}
            sample = nodes.common_ksampler(model=ip_model_patcher[0], 
                        seed=543543, 
                        steps=30, 
                        cfg=7.0, 
                        sampler_name="dpmpp_2m", 
                        scheduler="karras", 
                        positive=positive[0], 
                        negative=negative[0],
                        latent=latent, 
                        denoise=1)
        if latent_upscale:
            with torch.inference_mode():
                sample = sample[0]["samples"].to(torch.float16)
                self.vae.first_stage_model.cuda()
                decoded = self.vae.decode_tiled(sample).detach()
            print(torch.cuda.memory_cached(device=None))
            model_management.cleanup_models()
            gc.collect()
            model_management.soft_empty_cache()
            print(torch.cuda.memory_cached(device=None))
            final_image = Image.fromarray(np.array(decoded*255, dtype=np.uint8)[0])
            final_image.save("/content/final_image.png")
            latent_up = upscale(sample, 1.5, self.model_up, self.device, self.vae_device)
            sample_up = nodes.common_ksampler(model=ip_model_patcher[0], 
                                        seed=543543, 
                                        steps=30, 
                                        cfg=7.0, 
                                        sampler_name="dpmpp_2m", 
                                        scheduler="karras", 
                                        positive=positive[0], 
                                        negative=negative[0],
                                        latent=latent_up[0], 
                                        denoise=0.55)

            with torch.inference_mode():
                sample_up = sample_up[0]["samples"].to(torch.float16)
                self.vae.first_stage_model.cuda()
                decoded_up = self.vae.decode_tiled(sample_up).detach()
            print(torch.cuda.memory_cached(device=None))
            model_management.cleanup_models()
            gc.collect()
            model_management.soft_empty_cache()
            print(torch.cuda.memory_cached(device=None))
            final_up_image = Image.fromarray(np.array(decoded_up*255, dtype=np.uint8)[0])
            final_up_image.save("/content/final_up_image.png")
            return [Path("/content/final_image.png"), Path("/content/final_up_image.png")]
        else:
            with torch.inference_mode():
                sample = sample[0]["samples"].to(torch.float16)
                self.vae.first_stage_model.cuda()
                decoded = self.vae.decode_tiled(sample).detach()
            print(torch.cuda.memory_cached(device=None))
            model_management.cleanup_models()
            gc.collect()
            model_management.soft_empty_cache()
            print(torch.cuda.memory_cached(device=None))
            final_image = Image.fromarray(np.array(decoded*255, dtype=np.uint8)[0])
            final_image.save("/content/final_image.png")
            return [Path("/content/final_image.png")]