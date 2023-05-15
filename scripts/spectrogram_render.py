import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import librosa

import soundfile 
import time
import subprocess, os
from ProgressBar import *

################################################################################

SAVE_RENDER = 1
RENDER_STYLE = 'spec' # 'chroma' | 'spec'

samples_dir = '/home/jd/Documents/Projects/ames_music_visualizer/tracks'
audio_input_filename = samples_dir + '/' +'Everbloom_2-4-23_Master_2.wav'

# audio_input_filename = librosa.ex('trumpet') # Short sample sound

renders_dir = '/home/jd/Documents/Projects/ames_music_visualizer/renders'
render_num = 1
while os.path.isfile(renders_dir + '/' + 'av_sample_' + str(render_num)  + '.mp4'):
    render_num += 1

audio_output_filename = renders_dir + '/' +'audio_sample_' + str(render_num)  + '.wav'
video_output_filename = renders_dir + '/' +'video_sample_' + str(render_num) + '.mp4'
av_output_filename = renders_dir + '/' + 'av_sample_' + str(render_num)  + '.mp4'

sample_duration_sec_0 = 30 # 30.0 # [sec] Duration of track to sample, set = 0 to samnple entire track
sample_offset = 0.0          # [sec] Offest ionto track to betgin sample
sample_rate = 44100

fps = 30 # Rendered viualization framerate
dt = 1.0/fps # Delay between frames in seconds.
dt_ms = dt*1000.0; # Delay in milliseconds

num_hops_per_frame = 2

display_history_duration = 2.0 # [sec] Duration of spectrogram history visible in each rendered frame

#################################################################################

if sample_duration_sec_0 > 0:
    sample_data, sample_rate = librosa.load(audio_input_filename, mono=True, offset=sample_offset, duration=sample_duration_sec_0, sr=sample_rate)
else:
    sample_data, sample_rate = librosa.load(audio_input_filename, mono=True, offset=sample_offset, sr=sample_rate)

# sample_duration_sec = sample_total_frames*dt

sample_size = sample_data.size
sample_duration_sec = sample_size/sample_rate
sample_total_frames = int(sample_duration_sec*fps)

sample_hop_length = int(sample_size/(sample_total_frames*num_hops_per_frame)) 

sample_stft = librosa.stft(sample_data, hop_length=sample_hop_length)
sample_spectrogram = np.abs(sample_stft)
sample_spectrogram_dB = librosa.amplitude_to_db(sample_spectrogram, ref=np.max)

S = np.abs(librosa.stft(sample_data, n_fft=4096))**2
sample_chroma = librosa.feature.chroma_stft(S=S, sr=sample_rate)

#################################################################################

print('av_output_filename = ', av_output_filename)

print('frame_rate (fps) = ', fps)
print('sample_total_frames = ',sample_total_frames)
print('sample_duration_sec = ',sample_duration_sec)

print('sample_rate = ',sample_rate)
print('sample_size = ',sample_size)
print('sample_hop_length = ',sample_hop_length)

print('sample_spectrogram.shape = ', sample_spectrogram.shape)
print('sample_spectrogram_num_freq = ', sample_spectrogram.shape[0])
print('sample_spectrogram_num_frames = ', sample_spectrogram.shape[1])

#################################################################################

# fig, ax = plt.subplots(nrows=3, ncols=1) #, sharex=True)

# time_series_ax = ax[0]
# spectrogram_ax = ax[1]
# chroma_ax = ax[2]

# librosa.display.waveshow(sample_data, sr=sample_rate, ax=time_series_ax)
# spectrogram_full_img = librosa.display.specshow(sample_spectrogram_dB, y_axis='log', sr=sample_rate, x_axis='time', ax=spectrogram_ax, hop_length=sample_hop_length)
# chromagram_full_img = librosa.display.specshow(sample_chroma, y_axis='chroma', sr=sample_rate, x_axis='time', ax=chroma_ax, hop_length=sample_hop_length)

# plt.rcParams.update({'axes.titlesize': 8,'axes.labelsize': 8})

# plt.show()

# exit()

#################################################################################

display_history_num_hops = int(display_history_duration*fps*num_hops_per_frame)

print('display_history_duration = ', display_history_duration)
print('display_history_num_hops = ', display_history_num_hops)

px = 1/plt.rcParams['figure.dpi']  # pixel in inches
frame_fig, frame_ax = plt.subplots(figsize=(1920*px, 1080*px))


frame_fig.set_facecolor('black')
axis_color = '#808080'

frame_ax.yaxis.label.set_color(axis_color)
frame_ax.yaxis.label.set_fontsize('medium')

frame_ax.tick_params(axis='y', colors=axis_color, labelsize='medium', )
frame_ax.yaxis.grid(visible=True, color=axis_color, linewidth=0.5)

if RENDER_STYLE == 'chroma':

    frame_data_0 = sample_chroma[:,0:display_history_num_hops]
    frame_display = librosa.display.specshow(data=frame_data_0, x_axis='time', y_axis='chroma', sr=sample_rate, ax=frame_ax)

else:

    freq_start_idx = int(sample_spectrogram.shape[0]/2)
    freq_end_idx = int(sample_spectrogram.shape[0])

    frame_data_0 = sample_spectrogram_dB[freq_start_idx:freq_end_idx,0:display_history_num_hops]
    frame_display = librosa.display.specshow(data=frame_data_0, x_axis='time', y_axis='log', sr=sample_rate, ax=frame_ax)

# frame_fig.colorbar(frame_display, ax=frame_ax, format="%+2.f dB") # Show colorbar

# print('frame_data_0.shape=',np.shape(frame_data_0))

const_val = np.amin(frame_data_0) #[-1,0] # Value with which to pad spectrogram display array

def frame_update(frame_i):

    printProgressBar(frame_i, sample_total_frames, prefix = 'Progress:', suffix = 'Complete', length = 50)

    start_idx = np.maximum(0,frame_i-display_history_num_hops)

    if RENDER_STYLE == 'chroma':
        frame_data_i = sample_chroma[:,start_idx:frame_i]
    else:
        frame_data_i = sample_spectrogram_dB[freq_start_idx:freq_end_idx,start_idx:frame_i]

    pad_width = frame_data_0.shape[1] - frame_data_i.shape[1]

    if pad_width > 0:
        frame_data_i = np.pad(frame_data_i,((0,0),(pad_width,0)),'constant',constant_values=const_val)
        
    # print('frame_data_i.shape=',np.shape(frame_data_i))

    frame_display.set_array(frame_data_i)

    return

#################################################################################

ani = animation.FuncAnimation(frame_fig, frame_update,frames=sample_total_frames, repeat=False)

if  SAVE_RENDER:

    print("Rendering animation video ... ")

    # Write video animation file
    ani.save(video_output_filename, fps=fps)
    plt.show(block=False)
                    
    # Write audio sample file
    print('Saving audio sample track ...')
    soundfile.write(audio_output_filename, sample_data, sample_rate, format='wav', subtype='PCM_24')

    # Add audio track to video
    print('Adding audio track to video ...')
    subprocess.run(['ffmpeg', '-i', video_output_filename, '-i', audio_output_filename, '-c:v', 'copy', '-c:a', 'aac', av_output_filename, '-strict', '-2'])

    # Open rendered video in default system app
    subprocess.run(('xdg-open', av_output_filename))

else:

    print("Displaying (not saving) animation ... ")

    plt.show()

