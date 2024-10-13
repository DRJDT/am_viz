
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
from pathlib import Path
import numpy as np
import re
import json
import subprocess, os
import cv2

##########################################################################

# setlist = ['tiger_balm','peacock','groovin','unblinking_eye','hummin','walk_the_walk','the_governors_dead','cerulean_goodbye','stir_my_heart_awake']

setlist = ['walk_the_walk','the_governors_dead']

tracks_dir = "/home/jd/devel/am_viz/inputs/track_dirs/"

for track_name in setlist:

    track_dir = tracks_dir + track_name + '/'

    input_vid_file = track_dir + "source_vid_with_audio.mp4"
    extracted_frame_dir = track_dir + "extracted_video_frames/"

    audio_input_filename = track_dir + "extracted_audio_track.mp3"

    spectrogram_output_filename = track_dir + 'spectrogram_vid.mp4'
    spectrogram_with_audio_output_filename = track_dir + "spectrogram_vid_with_audio.mp4"

    generated_frame_dir = track_dir + "generated_frame_images/"
    generated_vid_filename = track_dir + 'generated_vid.avi'
    generated_vid_with_audio_filename = track_dir + 'generated_vid_with_audio.avi'

    Path(generated_frame_dir).mkdir(parents=True, exist_ok=True)

    sample_period_sec = 5.0
    n_steps_per_frame = 10
    generated_vid_frame_rate = n_steps_per_frame/sample_period_sec

   # GEN PARAMS: num_inference_steps = 40, strength = 0.2, text_prompt_weight = 0.5

    text_prompt_weight = 0.5

    min_strength = 0.1
    max_strength = 0.5

    ##########################################################################

    lyrics_data_file = track_dir + 'lyrics_data.json'

    # Writing to lyrics_data.json
    with open(lyrics_data_file) as json_file:
        json_data = json.load(json_file)
        lyrics_data = json_data['lyrics_data']
    
    # print(lyrics_data)

    n_input_frames = len(lyrics_data)

    # ##########################################################################

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
    # # print("Saving Initial Image: " + generated_frame_dir + img_filename)
    # # init_image.save(generated_frame_dir + img_filename)

    steps = np.linspace(min_strength,max_strength,n_steps_per_frame - 1)

    # ##########################################################################

    # frame_width = 512
    # frame_height = 512

    # generated_video = cv2.VideoWriter(generated_vid_filename, 0, generated_vid_frame_rate, (frame_width, frame_height))

    output_frame_i = 0

    for frame_i in range(n_input_frames):

        input_filename = "frame_" + str(frame_i) + ".png"

        init_image = load_image(extracted_frame_dir + input_filename).resize((512,512))

        output_filename = "frame_" + str(output_frame_i) + ".png"

        output_frame_i = output_frame_i + 1

        print("Saving Init Frame Image: " + generated_frame_dir + output_filename)
        init_image.save(generated_frame_dir + output_filename)

        ##########################################################################
        # text_prompt = lyrics[frame_i][0]

        text_prompt = ''

        if lyrics_data[frame_i]['use_lyrics_as_prompt']:
            text_prompt = text_prompt + lyrics_data[frame_i]['lyrics_phrase']

        if lyrics_data[frame_i]['use_aux_prompt']:
            text_prompt = text_prompt + ' ' + lyrics_data[frame_i]['aux_prompt']

        print("Frame text prompt: " + text_prompt)

        ##########################################################################

        last_image = init_image

        for step_i,si in enumerate(steps):

            image = pipeline(text_prompt, image=last_image, num_inference_steps = 40, strength = 0.2, guidance_scale=text_prompt_weight, generator=generator).images[0]

            last_image = image

            ##########################################################################

            output_filename = "frame_" + str(output_frame_i) + ".png"

            print("Saving Generated Frame Image: " + generated_frame_dir + output_filename)
            image.save(generated_frame_dir + output_filename)

            # generated_video.write(cv2.imread(generated_frame_dir + output_filename))

            output_frame_i = output_frame_i + 1

        # cv2.destroyAllWindows()
        # generated_video.release()

        # # Add audio track to video
        # print('Adding audio track to generated video ...')
        # subprocess.run(['ffmpeg', '-i', generated_vid_filename, '-i', audio_input_filename, '-c:v', 'copy', '-c:a', 'aac', generated_vid_with_audio_filename, '-strict', '-2'])