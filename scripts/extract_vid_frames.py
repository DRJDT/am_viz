import sys
import argparse
import cv2

print(cv2.__version__)

def extractImages(pathIn, pathOut, sample_delay_sec):
    sample_time_sec = 0
    frame_i = 0
    vidcap = cv2.VideoCapture(pathIn)
    # success,image = vidcap.read()

    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(sample_time_sec*1000))   
        success,image = vidcap.read()
        cv2.imwrite( pathOut + "frame_%d.png" % frame_i, image) 
        sample_time_sec = sample_time_sec + sample_delay_sec
        frame_i = frame_i + 1

if __name__=="__main__":


    input_vid_file = "inputs/tiger_balm/Tiger Balm Lyric Vid2.mp4"
    output_frame_dir = "inputs/tiger_balm/extracted_video_frames/"

    extractImages(input_vid_file, output_frame_dir, 5.0)