import torch
from diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler, LCMScheduler
from diffusers.utils import export_to_gif

adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
# pipe = AnimateDiffPipeline.from_pretrained(
#     "CompVis/stable-diffusion-v1-4", #"frankjoshua/toonyou_beta6", # "emilianJR/epiCRealism", #
#     motion_adapter=adapter, dtype=torch.float16,
# ).to("cuda")

# # set scheduler
# pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)


model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=torch.float16)
scheduler = DDIMScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)
pipe.scheduler = scheduler

# load LCM-LoRA
# pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5", adapter_name="lcm", torch_dtype=torch.float16)
# pipe.load_lora_weights("guoyww/animatediff-motion-lora-zoom-in", weight_name="diffusion_pytorch_model.safetensors", adapter_name="motion-lora", torch_dtype=torch.float16)

# pipe.set_adapters(["lcm", "motion-lora"], adapter_weights=[0.55, 1.2])

# pipe.set_adapters(["lcm"], adapter_weights=[1.0])

# enable memory savings
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

# pipe.unet.half()
# pipe.text_encoder.half()


# frames = pipe(
#     prompt=prompt,
#     negative_prompt="bad quality, worse quality",
#     num_frames=16,
#     guidance_scale=7.5,
#     num_inference_steps=25,
#     generator=torch.Generator("cpu").manual_seed(42),
# ).frames[0]






output = pipe(
    prompt=(
        "masterpiece, bestquality, highlydetailed, ultradetailed, sunset, "
        "orange sky, warm lighting, fishing boats, ocean waves seagulls, "
        "rippling water, wharf, silhouette, serene atmosphere, dusk, evening glow, "
        "golden hour, coastal landscape, seaside scenery"
    ),
    negative_prompt="bad quality, worse quality",
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=25,
    generator=torch.Generator("cpu").manual_seed(42),
)
frames = output.frames[0]
export_to_gif(frames, "data/fine_tune_datasets/animtatediff/animatelcm.gif")






# import torch
# from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
# from diffusers.utils import export_to_gif

# adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=torch.float16)
# pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=torch.float16)
# pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")

# pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora")
# pipe.set_adapters(["lcm-lora"], [0.8])

# pipe.enable_vae_slicing()
# pipe.enable_model_cpu_offload()

# output = pipe(
#     prompt="racoon riding dinosaur",
#     negative_prompt="",
#     num_frames=16,
#     guidance_scale=2.0,
#     num_inference_steps=6,
#     generator=torch.Generator("cpu").manual_seed(0),
# )
# frames = output.frames[0]
# export_to_gif(frames, "data/fine_tune_datasets/animtatediff/animatelcm.gif")