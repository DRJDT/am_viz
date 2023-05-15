from potentialflow import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import librosa
# from scipy import signal
import soundfile 
import time
import subprocess, os

################################################################################

samples_dir = '/home/jd/Documents/Projects/ames_music_visualizer/tracks/'
audio_input_filename = samples_dir + 'Everbloom_2-4-23_Master_2.wav'

# audio_input_filename = librosa.ex('trumpet') # Short sample sound

renders_dir = '/home/jd/Documents/Projects/ames_music_visualizer/renders/'
render_num = 1
while os.path.isfile(renders_dir + '/' + 'av_sample_' + str(render_num)  + '.mp4'):
    render_num += 1

audio_output_filename = renders_dir + '/' +'audio_sample_' + str(render_num)  + '.wav'
video_output_filename = renders_dir + '/' +'video_sample_' + str(render_num) + '.mp4'

av_output_filename = renders_dir + '/' + 'av_sample_' + str(render_num)  + '.mp4'

sample_offset = 0.0
sample_duration_sec_0 = 30.0
sample_rate = 44100

fps = 30 # Rendered viualization framerate
dt = 1.0/fps # Delay between frames in seconds.
dt_ms = dt*1000.0; # Delay in milliseconds

scaler_field_name = 'streamfunction' # ['streamfunction', 'potential', 'xvel', 'yvel', 'velmag', 'Cp']

flow_max_strength = 100.0
flow_smooth_frames = 5

n_contour_levels = 50
particle_gen_interval = 3 # [frames]

#################################################################################

sample_num_frames = int(sample_duration_sec_0*fps)
sample_duration_sec = sample_num_frames*dt

sample_data, sample_rate = librosa.load(audio_input_filename, mono=True, offset=sample_offset, duration=sample_duration_sec, sr=sample_rate)

sample_size = sample_data.size
sample_duration_sec_verify = sample_size/sample_rate

sample_hop_length = int(sample_size/sample_num_frames)     # 1024

fig, ax = plt.subplots(nrows=4, ncols=1) #, sharex=True)

flow_strength_ax = ax[0]
chroma_ax = ax[1]
spectrogram_ax = ax[2]
time_series_ax = ax[3]

sample_stft = librosa.stft(sample_data, hop_length=sample_hop_length)
sample_spectrogram = np.abs(sample_stft)
sample_amplitude_dB = librosa.amplitude_to_db(sample_spectrogram, ref=np.max)

S = np.abs(librosa.stft(sample_data, n_fft=4096))**2
sample_chroma = librosa.feature.chroma_stft(S=S, sr=sample_rate)

# spectrogram_ax.set(title='Log-frequency power spectrogram')
# spectrogram_ax.label_outer()
# fig.colorbar(img, ax=spectrogram_ax, format="%+2.f dB")

sample_envelope = np.sum(sample_spectrogram,0)
sample_envelope_dB = librosa.amplitude_to_db(sample_envelope, ref=np.max)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

flow_strength = sample_envelope_dB - np.min(sample_envelope_dB) # shift to zero min
flow_strength = flow_strength/np.max(flow_strength) 
# flow_strength = 2*flow_strength/np.max(flow_strength) - 1 # shift zero mean
flow_strength = flow_max_strength*flow_strength # scale to max strength
flow_strength = moving_average(flow_strength, flow_smooth_frames) # smooth with moving average

print('av_output_filename = ', av_output_filename)

print('frame_rate (fps) = ', fps)
print('sample_num_frames = ',sample_num_frames)
print('sample_duration_sec = ',sample_duration_sec)

print('sample_rate = ',sample_rate)
print('sample_size = ',sample_size)
print('sample_duration_sec_verify = ',sample_duration_sec_verify)
print('sample_hop_length = ',sample_hop_length)

print('sample_spectrogram.shape = ', sample_spectrogram.shape)
print('sample_spectrogram_num_freq = ', sample_spectrogram.shape[0])
print('sample_spectrogram_num_frames = ', sample_spectrogram.shape[1])
print('flow_strength.shape = ', flow_strength.shape)


librosa.display.waveshow(sample_data, sr=sample_rate, ax=time_series_ax)
img = librosa.display.specshow(sample_amplitude_dB, y_axis='log', sr=sample_rate, x_axis='time', ax=spectrogram_ax, hop_length=sample_hop_length)
img2 = librosa.display.specshow(sample_chroma, y_axis='chroma', sr=sample_rate, x_axis='time', ax=chroma_ax, hop_length=sample_hop_length)
flow_strength_ax.plot(flow_strength)

plt.rcParams.update({'axes.titlesize': 8,'axes.labelsize': 8})

plt.show()

# exit()

#################################################################################
# Potential Flow Simulation 

x_points = np.linspace(-10, 10, 100)
y_points = np.linspace(-10, 10, 100)

percentiles_to_include = 90

field_0 = Flowfield([
    Freestream(0.0,-10.0) # Freestream Flow 
])

X,Y,points = field_0.mesh_points(x_points,y_points)

scalar_field_0 = field_0.get_scalar(scaler_field_name, x_points, y_points)

# field_0.draw(scaler_field_name, x_points, y_points)

#################################################################################

# xp_0 = np.array([5.0, -5.0])
# yp_0 = np.array([-10.0, -10.0])

xp_0 = np.array([])
yp_0 = np.array([])

#################################################################################

xp = xp_0
yp = yp_0

n_pts = xp.size

#################################################################################

fig, ax = plt.subplots()

fig.patch.set_facecolor('black')
ax.set_axis_off()

# ax.set_facecolor("black")

# ax.grid(False)
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)

# fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)


scalar_min = np.nanpercentile(scalar_field_0, 50 - percentiles_to_include / 2)
scalar_max = np.nanpercentile(scalar_field_0, 50 + percentiles_to_include / 2)
contour_levels = np.linspace(scalar_min, scalar_max, n_contour_levels)

contour_kwargs = {
                "levels": n_contour_levels,
                "linewidths": 0.5,
                "alpha": 0.8,
                "cmap" : plt.get_cmap("cool") # cool, jet, plasma, ...
               }


c_data = ax.contour(X, Y, scalar_field_0.reshape(X.shape), **contour_kwargs)

particle_kwargs = {
                    "marker":'o', 
                    "markersize": 1, 
                    "color": "white",
                    "alpha": 0.3, 
                    "linestyle": "none",
                  }

particle_data, = ax.plot(xp,yp,**particle_kwargs)

# def init():
#     return plot_data

def frame_update(frame_i):

    global xp, yp, n_pts, particle_data, c_data

    # strength_i = 500*np.sin(2*np.pi*frame_i/cycle_duration_frames)+1

    strength_i = flow_strength[frame_i]
    complement_strength_i = flow_max_strength - strength_i

    alpha = strength_i/flow_max_strength*np.pi

    # print('strength_i = ', strength_i)

    field_i = Flowfield([
        Vortex(-strength_i,5,0),
        Vortex(strength_i,-5,0),
        Freestream(0,-10.0)
    ])

    #################################################################################

    saclar_field_i = field_i.get_scalar(scaler_field_name, x_points, y_points)

    # Geenrate random particle to track
    if (frame_i % particle_gen_interval) == 0:

        # random location
        xp = np.append(xp, np.random.random(1)*20 - 10)
        # yp = np.append(yp, np.random.random(1)*20 - 10)

        # xp = np.append(xp, 0)
        yp = np.append(yp, 10)

        n_pts += 1

    for pti in range(n_pts):

        x_vel = field_i.get_scalar("xvel", xp[pti], yp[pti])
        y_vel = field_i.get_scalar("yvel", xp[pti], yp[pti])

        if not(np.isfinite(x_vel) and np.isfinite(y_vel)):
            print("vel infinite")
            continue
        else:

            xp[pti] += x_vel*dt
            yp[pti] += y_vel*dt

    particle_data.set_data(xp, yp)
    
    # Clear contour data
    for coll in c_data.collections:
        coll.remove()

    # replot frame contours
    c_data = ax.contour(X, Y, saclar_field_i.reshape(X.shape), **contour_kwargs)

    return particle_data

#################################################################################

print("Rendering animation video ... ")
ani = animation.FuncAnimation(fig, frame_update, frames=sample_num_frames, repeat=False)
                    
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

