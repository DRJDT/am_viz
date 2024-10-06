import os
import cv2

import subprocess, os
import re


frame_image_dir= "outputs/tiger_balm/generated_frame_images_4/"
video_file = "outputs/tiger_balm/generated_video_4.avi"
output_av_file = "outputs/tiger_balm/generated_audio_video_4.avi"

audio_file = "inputs/tiger_balm/track.mp3"

frame_extraction_period_sec= 5.0
n_steps_per_frame = 10
frame_rate = n_steps_per_frame/frame_extraction_period_sec

image_files = os.listdir(frame_image_dir)

image_files_sorted = {}

for image_filename in image_files:

    idx = re.compile(r'_(\d+)').search(image_filename).group(1)

    image_files_sorted[int(idx)] = image_filename


frame_0 = cv2.imread(frame_image_dir + image_files_sorted[0])   

height, width, layers = frame_0.shape

video = cv2.VideoWriter(video_file, 0, frame_rate, (width, height))

for frame_i in range(len(image_files_sorted)): 

    image_file_i = image_files_sorted[frame_i]

    print("writing frame: " + image_file_i) 

    video.write(cv2.imread(os.path.join(frame_image_dir, image_file_i)))

cv2.destroyAllWindows()
video.release()


# Add audio track to video
print('Adding audio track to video ...')
subprocess.run(['ffmpeg', '-i', video_file, '-i', audio_file, '-c:v', 'copy', '-c:a', 'aac', output_av_file, '-strict', '-2'])