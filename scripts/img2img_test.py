
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
from pathlib import Path
import numpy as np

##########################################################################

output_dir = "outputs/tiger_balm/"

init_image = load_image("inputs/tiger_balm/vid_frames/vid_frame_1.png").resize((512,512))

text_prompt = "from my lungs into the speakers"

##########################################################################

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

##########################################################################

img_filename = "img2img" + "_frame_0" + ".png"
print("Saving Initial Image: " + output_dir + img_filename)
init_image.save(output_dir + img_filename)

text_prompt_weight = 1.0

N_steps = 5
min_strength = 0.1
max_strength = 0.5

steps = np.linspace(min_strength,max_strength,N_steps)

for i,si in enumerate(steps):

    image = pipeline(text_prompt, image=init_image, num_inference_steps=20, strength=si, guidance_scale=text_prompt_weight, generator=generator).images[0]

    ##########################################################################

    img_filename = "img2img" + "_frame_" + str(i+1) + ".png"

    print("Saving Generated Image: " + output_dir + img_filename)
    image.save(output_dir + img_filename)


