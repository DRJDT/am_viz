import sys
import argparse
import cv2
from pathlib import Path
import subprocess, os

print(cv2.__version__)

window_name = 'image'

def extractImages(pathIn, pathOut, sample_period_sec):

    sample_start_sec = 0.0
    frame_i = 0

    vidcap = cv2.VideoCapture(pathIn)
    # success,image = vidcap.read()

    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(sample_start_sec*1000))   
        success,image = vidcap.read()

        if success:

            # cv2.imshow(window_name,image)
            # cv2.waitKey(0)

            print('Extracting frame from video ...' + str(frame_i) )
            cv2.imwrite( pathOut + "frame_%d.png" % frame_i, image) 

        sample_start_sec = sample_start_sec + sample_period_sec
        frame_i = frame_i + 1

if __name__=="__main__":

    # setlist = ['tiger_balm','peacock','groovin','unblinking_eye','hummin','walk_the_walk','the_governors_dead','cerulean_goodbye','stir_my_heart_awake']

    setlist = ['walk_the_walk']

    tracks_dir = "/home/jd/devel/am_viz/inputs/track_dirs/"

    # track_name = "groovin"

    for track_name in setlist:

        track_dir = tracks_dir + track_name + '/'

        input_vid_file = track_dir + "source_vid_with_audio.mp4"

        extracted_frame_dir = track_dir + "extracted_video_frames/"

        output_audio_file = track_dir + "extracted_audio_track.mp3"

        ###################################################################

        Path(extracted_frame_dir).mkdir(parents=True, exist_ok=True)

        # Extract audio track from video
        print('Extracting audio track from video ...')
        subprocess.run(['ffmpeg', '-i', input_vid_file, '-b:a', '192K', '-vn', output_audio_file,'-y'])

        # Extract video frames at specified interval
        sample_period_sec = 5.0

        extractImages(input_vid_file, extracted_frame_dir, sample_period_sec)