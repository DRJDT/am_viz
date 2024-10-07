import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import librosa

import soundfile 
import time
import subprocess, os
# from alive_progress import alive_bar

###############################################################################

SAVE_RENDER = 1
RENDER_STYLE = 'spec' # 'chroma' | 'spec'

samples_dir = '/home/jd/Software/ames_ai_viz/inputs/tiger_balm/'
audio_input_filename = '/home/jd/Software/ames_ai_viz/inputs/tiger_balm/track.mp3'

renders_dir = '/home/jd/Software/ames_ai_viz/outputs/tiger_balm/'
render_num = 1
while os.path.isfile(renders_dir + '/' + 'spectrogram_' + str(render_num)  + '.mp4'):
    render_num += 1

audio_output_filename = renders_dir + '/' +'spectrogram_track_' + str(render_num)  + '.wav'
video_output_filename = renders_dir + '/' +'spectrogram_video_' + str(render_num) + '.mp4'
av_output_filename = renders_dir + '/' + 'spectrogram_' + str(render_num)  + '.mp4'

fps = 2.0 # Rendered visualization framerate [default: 30]
dt = 1.0/fps # Delay between frames in seconds.
dt_ms = dt*1000.0; # Delay in milliseconds

num_hops_per_frame = 64 #

sample_duration_sec_0 = 0 # [sec] Duration of track to sample, set = 0 to sample entire track
sample_offset = 0.0          # [sec] Offset into track to begin sample
sample_rate = 44100

#################################################################################

if sample_duration_sec_0 > 0:
    track_data, track_rate = librosa.load(audio_input_filename, mono=True, offset=sample_offset, duration=sample_duration_sec_0, sr=sample_rate)
else:
    track_data, track_rate = librosa.load(audio_input_filename, mono=True, offset=sample_offset, sr=sample_rate)

num_samples = track_data.size
track_duration_sec = num_samples/sample_rate
num_frames = int(track_duration_sec*fps)

samples_per_hop = int(num_samples/(num_hops_per_frame*num_frames)) 

#################################################################################

# fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)

# time_series_ax = ax[0]
# spectrogram_ax = ax[1]
# chroma_ax = ax[2]

# librosa.display.waveshow(sample_data, sr=sample_rate, ax=time_series_ax, color='m')
# spectrogram_full_img = librosa.display.specshow(sample_spectrogram_dB, y_axis='log', sr=sample_rate, x_axis='time', ax=spectrogram_ax, hop_length=sample_hop_length)
# chromagram_full_img = librosa.display.specshow(sample_chroma, y_axis='chroma', sr=sample_rate, x_axis='time', ax=chroma_ax, hop_length=sample_hop_length)

# plt.rcParams.update({'axes.titlesize': 8,'axes.labelsize': 8})

# time_series_ax.set_xlim([0,sample_duration_sec_0])
# time_series_ax.set_facecolor('k')

# plt.show()

# exit()

#################################################################################

px = 1/plt.rcParams['figure.dpi']  # pixel in inches
# frame_fig, frame_ax = plt.subplots(figsize=(1920*px, 1080*px))
frame_fig, frame_ax = plt.subplots(figsize=(512*px, 512*px))

frame_fig.set_facecolor('black')
axis_color = '#808080'

# frame_ax.yaxis.label.set_color(axis_color)
# frame_ax.yaxis.label.set_fontsize('medium')

# frame_ax.tick_params(axis='y', colors=axis_color, labelsize='medium', )
# frame_ax.yaxis.grid(visible=True, color=axis_color, linewidth=0.5)

#################

def compute_frame_spec(frame_i):

    sample_data, frame_sample_rate = librosa.load(audio_input_filename, mono=True, offset=frame_i*dt, duration=dt, sr=sample_rate)

    sample_stft = librosa.stft(sample_data, hop_length=samples_per_hop)
    sample_spectrogram = np.abs(sample_stft)
    sample_spectrogram_dB = librosa.amplitude_to_db(sample_spectrogram, ref=np.max)

    # S = np.abs(librosa.stft(sample_data, n_fft=4096))**2    # n_fft=4096
    # sample_chroma = librosa.feature.chroma_stft(S=S, sr=sample_rate)

    return sample_spectrogram_dB[20:,:] #  sample_chroma

#################

frame_data_0 = compute_frame_spec(0)

frame_display = librosa.display.specshow(data=frame_data_0, x_axis='time', y_axis='log', sr=sample_rate, ax=frame_ax)

frame_display.set_cmap("plasma")

def frame_update(frame_i):

    frame_data_i = compute_frame_spec(frame_i)

    frame_display.set_array(frame_data_i)

    print(str(frame_i) + '/' + str(num_frames))

    return

#################################################################################

ani = animation.FuncAnimation(frame_fig, frame_update, frames=num_frames, repeat=False)

if  SAVE_RENDER:

    print("Rendering animation video ... ")

    # Write video animation file
    ani.save(video_output_filename, fps=fps)
    plt.show(block=False)
        
    # Write audio sample file
    print('Saving audio sample track ...')
    soundfile.write(audio_output_filename, track_data, track_rate, format='wav', subtype='PCM_24')

    # Add audio track to video
    print('Adding audio track to video ...')
    subprocess.run(['ffmpeg', '-i', video_output_filename, '-i', audio_output_filename, '-c:v', 'copy', '-c:a', 'aac', av_output_filename, '-strict', '-2'])

    # Delete temp video and audio track files
    print('Cleaning up temp files ...')
    os.remove(audio_output_filename)
    os.remove(video_output_filename)

    # Open rendered video in default system app
    subprocess.run(('xdg-open', av_output_filename))

else:

    print("Displaying (not saving) animation ... ")

    plt.show()

