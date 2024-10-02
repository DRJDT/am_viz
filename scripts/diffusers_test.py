from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, UniPCMultistepScheduler


import torch
from PIL import Image
import numpy as np
import tqdm
from IPython.display import display

###############################################################################

output_dir = "/home/jd/Software/diffusers/outputs/"

prompt = "photograph of an astronaut riding a horse"

img_path = output_dir + "_".join(prompt.split()) 

repo_id = "CompVis/stable-diffusion-v1-4" # "google/ddpm-cat-256" #  # "CompVis/ldm-text2im-large-256"

#################################################################################

vae = AutoencoderKL.from_pretrained(repo_id, subfolder="vae", use_safetensors=True)
tokenizer = CLIPTokenizer.from_pretrained(repo_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(repo_id, subfolder="text_encoder", use_safetensors=True)
unet = UNet2DConditionModel.from_pretrained(repo_id, subfolder="unet", use_safetensors=True)
scheduler = UniPCMultistepScheduler.from_pretrained(repo_id, subfolder="scheduler")

################################################################################

torch_device = "cpu" # "cuda"
vae.to(torch_device)
text_encoder.to(torch_device)
unet.to(torch_device)

################################################################################

height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion
num_inference_steps = 25  # Number of denoising steps
guidance_scale = 7.5  # Scale for classifier-free guidance
generator = torch.manual_seed(0)  # Seed generator to create the initial latent noise
batch_size = 1

################################################################################

def save_sample(sample, img_path1):

    # image_processed = sample.cpu().permute(0, 2, 3, 1)

    # image_processed = (image_processed + 1.0) * 127.5

    # image_processed = image_processed.numpy().astype(np.uint8)

    # image_pil = Image.fromarray(image_processed[0])

    image = (sample / 2 + 0.5).clamp(0, 1).squeeze()
    image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    image = Image.fromarray(image)

    print("Saving Generated Image: " + img_path1)

    image.save(img_path1)

# ###############################################################################

# # pipeline = DiffusionPipeline.from_pretrained(repo_id)
# # pipeline.to("cpu")
# # image = pipeline(prompt).images[0]

###############################################################################
    
text_input = tokenizer(
    [prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
)

with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

###############################################################################

max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

###############################################################################
# text2img pipeline

# latents = torch.randn(
#     (batch_size, unet.config.in_channels, height // 8, width // 8),
#     generator=generator,
#     device=torch_device,
# )

# latents = latents * scheduler.init_noise_sigma

# ###############################################################################

# scheduler.set_timesteps(num_inference_steps)

# for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):

#     # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
#     latent_model_input = torch.cat([latents] * 2)

#     latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

#     # predict the noise residual
#     with torch.no_grad():
#         noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

#     # perform guidance
#     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

#     # compute the previous noisy sample x_t -> x_t-1
#     latents = scheduler.step(noise_pred, t, latents).prev_sample

#     # scale and decode the image latents with vae
#     with torch.no_grad():
#         sample_image = vae.decode((1 / 0.18215) * latents).sample

#     save_sample(sample_image, img_path + "_step_" + str(i) + ".png")


###############################################################################

###############################################################################

# print("Saving Generated Image: " + img_path + ".png")
# image.save(img_path + ".png")
    


# latents = torch.randn(
#     (batch_size, unet.config.in_channels, height // 8, width // 8),
#     generator=generator,
#     device=torch_device,
# )

# latents = latents * scheduler.init_noise_sigma

# ###############################################################################
# # img2img pipeline

# scheduler.set_timesteps(num_inference_steps)

# for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):

#     # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
#     latent_model_input = torch.cat([latents] * 2)

#     latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

#     # predict the noise residual
#     with torch.no_grad():
#         noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

#     # perform guidance
#     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

#     # compute the previous noisy sample x_t -> x_t-1
#     latents = scheduler.step(noise_pred, t, latents).prev_sample

#     # scale and decode the image latents with vae
#     with torch.no_grad():
#         sample_image = vae.decode((1 / 0.18215) * latents).sample

#     save_sample(sample_image, img_path + "_step_" + str(i) + ".png")