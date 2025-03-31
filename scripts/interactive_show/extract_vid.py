import sys
import argparse
import cv2
from pathlib import Path
import os

print(cv2.__version__)

window_name = 'image'

def extractImages(fileIn, dirOut, sample_period_sec):

    sample_start_sec = 0.0
    frame_i = 0

    vidcap = cv2.VideoCapture(fileIn)
    # success,image = vidcap.read()

    print('Extracting frame from video:' + fileIn)

    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(sample_start_sec*1000))   
        success,image = vidcap.read()

        fileOut = os.path.join(dirOut,'%s_frame_%d.png' % (os.path.basename(fileIn), frame_i))

        if success:

            # cv2.imshow(window_name,image)
            # cv2.waitKey(0)

            print('Extracting frame ...' + str(frame_i) )
            cv2.imwrite(fileOut, image) 

        sample_start_sec = sample_start_sec + sample_period_sec
        frame_i = frame_i + 1

if __name__=="__main__":

        parser = argparse.ArgumentParser(prog="extract_vid_frames",
                                         description="Extract frames of video and save as image files at defined rate.")

        parser.add_argument('-i', '--vid_input_dir',  
                            help="Directory containing input video files",
                            required=True)
        parser.add_argument('-o', '--img_output_dir', 
                            help="Directory containing output image files",
                            required=False,
                            default='')
        parser.add_argument('-f', '--frame_rate',     
                            help="Frame sample rate [hz].", 
                            type=float, 
                            default=1.0,
                            required=False)

        args = parser.parse_args()

        vid_input_dir = args.vid_input_dir
        img_output_dir = args.img_output_dir
        sample_rate = 1.0/args.frame_rate

        if not img_output_dir:
             img_output_dir = os.path.join(vid_input_dir,'extracted_frame_imgs','')
             Path(img_output_dir).mkdir(parents=True, exist_ok=True)

        ###################################################################

        vid_files = []
        for (dirpath, dirnames, filenames) in os.walk(vid_input_dir):
            vid_files.extend(filenames)
            break

        ###################################################################

        for vid_i in vid_files:

            # img_out_file = os.path.join(img_output_dir,os.path.basename(vid_file_i) + '_' + 

            fileIn = os.path.join(vid_input_dir,vid_i)

            extractImages(fileIn,img_output_dir, sample_rate)

        ###################################################################