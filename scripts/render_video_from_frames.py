import os
import cv2

import subprocess, os
import re

# setlist = ['tiger_balm','peacock','groovin','unblinking_eye','hummin','walk_the_walk','the_governors_dead','cerulean_goodbye','stir_my_heart_awake']

setlist = ['cerulean_goodbye']

tracks_dir = "/home/jd/devel/am_viz/inputs/track_dirs/"

for track_name in setlist:

    track_dir = tracks_dir + track_name + '/'

    frame_image_dir= track_dir + "generated_frame_images/"
    audio_file =  track_dir + "extracted_audio_track.mp3"
    video_file =  track_dir + "generated_vid.avi"
    output_av_file =  track_dir + "generated_vid_with_audio.avi"

    frame_extraction_period_sec= 5.0
    n_steps_per_frame = 10
    frame_rate = n_steps_per_frame/frame_extraction_period_sec

    image_files = os.listdir(frame_image_dir)

    # print(image_files)

    image_files_sorted = {}

    for image_filename in image_files:

        idx = re.compile(r'_(\d+)').search(image_filename).group(1)

        image_files_sorted[int(idx)] = image_filename

    frame_0 = cv2.imread(frame_image_dir + image_files_sorted[0])   

    # window_name = 'image'
    # cv2.imshow(window_name,frame_0)
    # cv2.waitKey(0)
    # cv2.imshow()
    # exit()

    height, width, layers = frame_0.shape

    # print(frame_0.shape)

    video = cv2.VideoWriter(video_file, 0, frame_rate, (width, height))

    for frame_i in range(len(image_files_sorted)): 

        image_file_i = image_files_sorted[frame_i]

        print("writing frame: " + image_file_i + "to file: " + video_file) 

        video.write(cv2.imread(os.path.join(frame_image_dir, image_file_i)))

    cv2.destroyAllWindows()
    video.release()


    # Add audio track to video
    print('Adding audio track to video ...')
    subprocess.run(['ffmpeg', '-i', video_file, '-i', audio_file, '-c:v', 'copy', '-c:a', 'aac', output_av_file, '-strict', '-2'])