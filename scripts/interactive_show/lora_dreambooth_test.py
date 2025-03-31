import os
import torch

from diffusers import StableDiffusionPipeline
# from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from peft import PeftModel, LoraConfig

# PRETRAINED_MODEL_NAME = "CompVis/stable-diffusion-v1-4" 
PRETRAINED_MODEL_NAME = "stable-diffusion-v1-5/stable-diffusion-v1-5"

# PRETRAINED_MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"

MODEL_DIR = "/home/jd/src/am_viz/data/fine_tune_datasets/"
MODEL_NAME = "ames-blue-flower-v1-5"

def get_lora_sd_pipeline(
    ckpt_dir, base_model_name_or_path=None, dtype=torch.float16, device="cuda", adapter_name="default"
):
    unet_sub_dir = os.path.join(ckpt_dir, "unet")
    text_encoder_sub_dir = os.path.join(ckpt_dir, "text_encoder")
    if os.path.exists(text_encoder_sub_dir) and base_model_name_or_path is None:
        config = LoraConfig.from_pretrained(text_encoder_sub_dir)
        base_model_name_or_path = config.base_model_name_or_path

    if base_model_name_or_path is None:
        raise ValueError("Please specify the base model name or path")

    pipe = StableDiffusionPipeline.from_pretrained(base_model_name_or_path, torch_dtype=dtype).to(device)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_sub_dir, adapter_name=adapter_name)

    if os.path.exists(text_encoder_sub_dir):
        pipe.text_encoder = PeftModel.from_pretrained(
            pipe.text_encoder, text_encoder_sub_dir, adapter_name=adapter_name
        )

    if dtype in (torch.float16, torch.bfloat16):
        pipe.unet.half()
        pipe.text_encoder.half()

    pipe.to(device)
    return pipe

if __name__ == "__main__":

    pipe = get_lora_sd_pipeline(os.path.join(MODEL_DIR,MODEL_NAME,"model"), PRETRAINED_MODEL_NAME)

    prompt = MODEL_NAME + " flowers bouquet"

    negative_prompt = "blurry, unfinished"

    image = pipe(prompt, num_inference_steps=50, guidance_scale=7, negative_prompt=negative_prompt).images[0]
    image.save(os.path.join(MODEL_DIR,MODEL_NAME,"test_output_imgs","test_1.png"))