import torch
import requests
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
# import numpy as np
from torchvision import transforms as tfms
from diffusers import StableDiffusionPipeline, DDIMScheduler


# Useful function for later
def load_image(url, size=None):
    response = requests.get(url, timeout=0.2)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    if size is not None:
        img = img.resize(size)
    return img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pipeline
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)

# Set up a DDIM scheduler
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)


# Sample function (regular DDIM)
@torch.no_grad()
def sample(
    prompt,
    start_step=0,
    start_latents=None,
    guidance_scale=3.5,
    num_inference_steps=50,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
    device=device,
):

    # Encode prompt
    text_embeddings = pipe._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Create a random starting point if we don't have one already
    if start_latents is None:
        start_latents = torch.randn(1, 4, 64, 64, device=device)
        start_latents *= pipe.scheduler.init_noise_sigma

    latents = start_latents.clone()

    for i in tqdm(range(start_step, num_inference_steps)):

        t = pipe.scheduler.timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Normally we'd rely on the scheduler to handle the update step:
        # latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # Instead, let's do it ourselves:
        prev_t = max(1, t.item() - (1000 // num_inference_steps))  # t-1
        alpha_t = pipe.scheduler.alphas_cumprod[t.item()]
        alpha_t_prev = pipe.scheduler.alphas_cumprod[prev_t]
        predicted_x0 = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        direction_pointing_to_xt = (1 - alpha_t_prev).sqrt() * noise_pred
        latents = alpha_t_prev.sqrt() * predicted_x0 + direction_pointing_to_xt

    # Post-processing
    images = pipe.decode_latents(latents)
    images = pipe.numpy_to_pil(images)

    return images

# Sample function (regular DDIM)
@torch.no_grad()
def encode_image(
    input_image,
    device=device,
):

    latent = pipe.vae.encode(tfms.functional.to_tensor(input_image).unsqueeze(0).to(device) * 2 - 1)
    return 0.18215 * latent.latent_dist.sample()


## Inversion
@torch.no_grad()
def invert(
    start_latents,
    prompt,
    guidance_scale=3.5,
    num_inference_steps=50,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
    device=device,
):

    # Encode prompt
    text_embeddings = pipe._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    # Latents are now the specified start latents
    latents = start_latents.clone()

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = []

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
    timesteps = reversed(pipe.scheduler.timesteps)

    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps - 1):

        # We'll skip the final iteration
        if i >= num_inference_steps - 1:
            continue

        t = timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
        next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]

        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
            1 - alpha_t_next
        ).sqrt() * noise_pred

        # Store
        intermediate_latents.append(latents)

    return torch.cat(intermediate_latents)

#########################################################################################################################

if  __name__ == "__main__":

    # # Test our sampling function by generating an image
    # sample("Watercolor painting of a beach sunset", negative_prompt=negative_prompt, num_inference_steps=50)[0].resize(
    #     (256, 256)
    # )

    num_inference_steps = 50
    start_step = 20

    guidance_scale = 3.5

    input_image = load_image("https://images.pexels.com/photos/8306128/pexels-photo-8306128.jpeg", size=(512, 512))

    plt.figure(1)
    plt.imshow(input_image)

    input_image_prompt = "Photograph of a puppy on the grass"
    inverted_image_prompt = "Photograph of a capybara on the grass"

    input_image_latents = encode_image(input_image)

    # # Plot 'alpha' (alpha_bar in DDPM language, alphas_cumprod in Diffusers for clarity)
    # timesteps = pipe.scheduler.timesteps.cpu()
    # alphas = pipe.scheduler.alphas_cumprod[timesteps]
    # # plt.plot(timesteps, alphas, label="alpha_t")
    # # plt.legend()

    inverted_latents = invert(input_image_latents, input_image_prompt, num_inference_steps=num_inference_steps)

    print("inverted_latents.shape=",inverted_latents.shape)

    # Decode the final inverted latents

    with torch.no_grad():

        start_latent_image = pipe.numpy_to_pil(pipe.decode_latents(inverted_latents[-1].unsqueeze(0)))[0]

        # inverted_images = sample(
        #     inverted_image_prompt, 
        #     start_latents=inverted_latents[-(start_step + 1)][None], 
        #     start_step=start_step,
        #     num_inference_steps=num_inference_steps, 
        #     guidance_scale=guidance_scale)

        input_latent_images = []
        for i in range(len(inverted_latents)):
            input_latent_images.append(pipe.numpy_to_pil(pipe.decode_latents(inverted_latents[-i-1].unsqueeze(0)))[0])
        
    plt.figure(2)
    plt.imshow(start_latent_image)

    # plt.figure(3)
    # plt.imshow(inverted_images[0])

    # Create a figure and a grid of subplots
    plt.figure(3)
    fig, axes = plt.subplots(nrows=5, ncols=10)
    plt.axis('off')

    axes = axes.flatten()                     
    for i in range(len(input_latent_images)):

        axes[i].imshow(input_latent_images[i])
        axes[i].set_axis_off()


    plt.show()

