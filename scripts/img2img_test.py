from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder", use_safetensors=True # torch_dtype=torch.float16
)
pipeline.to("cpu")

# pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
# pipeline.enable_xformers_memory_efficient_attention()

init_image = load_image("inputs/tiger_balm/vid_frames/vid_frame_1.png")

text_prompt = "from my lungs into the speakers"

# pass prompt and image to pipeline
image = pipeline(text_prompt, image=init_image, strength=0.8).images[0]


# make_image_grid([init_image, image], rows=1, cols=2)

print("Saving Generated Image: " + "img2img" + "_frame_1" + ".png")
image.save('outputs/tiger_balm' + "img2img" + "_frame_1" + ".png")