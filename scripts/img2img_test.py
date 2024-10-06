
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
from pathlib import Path
import numpy as np
import re

##########################################################################

_sec = 5.0

input_frame_dir = "inputs/tiger_balm/extracted_video_frames/"

output_dir = "outputs/tiger_balm/generated_frame_images_4/"

##########################################################################

lyrics_frame_sync_file = "inputs/tiger_balm/lyrics_frame_sync.txt"

lyrics = []

with open(lyrics_frame_sync_file) as file:

    for line in file:

        lyrics.append(line.split('"')[1::2])

n_input_frames = len(lyrics)

# ##########################################################################

Path(output_dir).mkdir(parents=True, exist_ok=True)

torch.backends.cuda.matmul.allow_tf32 = True

model = "kandinsky-community/kandinsky-2-2-decoder"
# model = "nota-ai/bk-sdm-small" # destilled
# model = "runwayml/stable-diffusion-v1-5"
vae = "sayakpaul/taesd-diffusers"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##########################################################################

# pipeline = StableDiffusionPipeline.from_pretrained(
#     model, torch_dtype=torch.float16, use_safetensors=True,
# ).to(device)

generator = torch.manual_seed(2023)
# image = pipeline(text_prompt, num_inference_steps=25, generator=generator).images[0]

##########################################################################

pipeline = AutoPipelineForImage2Image.from_pretrained(
    model, torch_dtype=torch.float16, use_safetensors=True
).to(device)

pipeline.vae = AutoencoderTiny.from_pretrained(
    vae, torch_dtype=torch.float16, use_safetensors=True,
).to(device)

pipeline.enable_model_cpu_offload()

pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)

# ##########################################################################

# # img_filename = "img2img" + "_frame_0" + ".png"
# # print("Saving Initial Image: " + output_dir + img_filename)
# # init_image.save(output_dir + img_filename)

text_prompt_weight = 1.0

N_steps = 9
min_strength = 0.1
max_strength = 0.5

steps = np.linspace(min_strength,max_strength,N_steps)

# ##########################################################################

output_frame_i = 0

for frame_i in range(n_input_frames):

    input_filename = "frame_" + str(frame_i) + ".png"

    init_image = load_image(input_frame_dir + input_filename).resize((512,512))

    output_filename = "frame_" + str(output_frame_i) + ".png"

    output_frame_i = output_frame_i + 1

    print("Saving Init Frame Image: " + output_dir + output_filename)
    init_image.save(output_dir + output_filename)

    text_prompt = lyrics[frame_i][0]
    print("Frame text prompt: " + text_prompt)

    last_image = init_image

    for step_i,si in enumerate(steps):

        image = pipeline(text_prompt, image=last_image, num_inference_steps = 10, strength = 0.2, guidance_scale=text_prompt_weight, generator=generator).images[0]

        last_image = image

        ##########################################################################

        output_filename = "frame_" + str(output_frame_i) + ".png"

        print("Saving Generated Frame Image: " + output_dir + output_filename)
        image.save(output_dir + output_filename)

        output_frame_i = output_frame_i + 1