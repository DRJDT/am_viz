from potentialflow import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import librosa
# from scipy import signal
import soundfile 
import time
import subprocess, os

################################################################################

samples_dir = '/home/jd/Documents/Miscellaneous/ames_music_visualizer/samples/'
audio_input_filename = samples_dir + 'Everbloom_2-4-23_Master_2.wav'

# audio_input_filename = librosa.ex('trumpet') # Short sample sound

renders_dir = '/home/jd/Documents/Miscellaneous/ames_music_visualizer/renders/'
render_num = 1

audio_output_filename = renders_dir + 'audio_sample_' + str(render_num)  + '.wav'
video_output_filename = renders_dir + 'video_sample_' + str(render_num) + '.mp4'
av_output_filename = renders_dir + 'av_sample_' + str(render_num)  + '.mp4'

sample_offset = 5.0
sample_duration_sec_0 = 15.0
sample_rate = 44100

fps = 30 # Rendered viualization framerate
dt = 1.0/fps # Delay between frames in seconds.
dt_ms = dt*1000.0; # Delay in milliseconds

#################################################################################

sample_num_frames = int(sample_duration_sec_0*fps)
sample_duration_sec = sample_num_frames*dt

sample_data, sample_rate = librosa.load(audio_input_filename, mono=True, offset=sample_offset, duration=sample_duration_sec, sr=sample_rate)

sample_size = sample_data.size
sample_duration_sec_verify = sample_size/sample_rate

sample_hop_length = int(sample_size/sample_num_frames)     # 1024

fig, ax = plt.subplots(nrows=3, ncols=1) #, sharex=True)
time_series_ax = ax[2]
spectrogram_ax = ax[1]
envelope_db_ax = ax[0]

sample_spectrogram = np.abs(librosa.stft(sample_data, hop_length=sample_hop_length))
sample_amplitude_dB = librosa.amplitude_to_db(sample_spectrogram, ref=np.max)

# spectrogram_ax.set(title='Log-frequency power spectrogram')
# spectrogram_ax.label_outer()
# fig.colorbar(img, ax=spectrogram_ax, format="%+2.f dB")

sample_envelope_dB = librosa.amplitude_to_db(np.sum(sample_spectrogram,0), ref=np.max)

sample_envelope_dB = sample_envelope_dB - np.min(sample_envelope_dB)

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
print('sample_envelope_dB.shape = ', sample_envelope_dB.shape)

# librosa.display.waveshow(sample_data, sr=sample_rate, ax=time_series_ax)
# img = librosa.display.specshow(sample_amplitude_dB, y_axis='log', sr=sample_rate, x_axis='time', ax=spectrogram_ax, hop_length=sample_hop_length)
# envelope_db_ax.plot(sample_envelope_dB)
# plt.show()

# exit()

#################################################################################

x_points = np.linspace(-10, 10, 200)
y_points = np.linspace(-10, 10, 100)
percentiles_to_include = 99.5

field_0 = Flowfield([
    Vortex(500,5.0,0.0),
    Vortex(-500,-5.0,0.0),
    # Vortex(500,5.0,5.0),
    # Vortex(500,-5.0,-5.0),
    # Vortex(-500,5.0,-5.0),
    # Vortex(-500,-5.0,5.0),
    # Source(100,5.0,0.0),
    # Source(-100,-5.0,0.0),
    # Freestream(0.0,5.0) # Freestream Flow Up Slowly (lavalamp)
])

X,Y,points = field_0.mesh_points(x_points,y_points)

streamfunction_0 = field_0.get_scalar("streamfunction", x_points, y_points)

# field_0.draw("streamfunction", x_points, y_points)

# xp_0 = np.array([5.0, -5.0])
# yp_0 = np.array([-10.0, -10.0])

xp_0 = np.array([5.0, -5.0, 1.0, 1.0])
yp_0 = np.array([1.0, 1.0, 5.0, -5.0])

xp = xp_0
yp = yp_0

n_pts = xp.size


# ################################################################################

fig, ax = plt.subplots()

ax.set_facecolor("black")
ax.grid(False)

ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

scalar_min = np.nanpercentile(streamfunction_0, 50 - percentiles_to_include / 2)
scalar_max = np.nanpercentile(streamfunction_0, 50 + percentiles_to_include / 2)
counter_levels = np.linspace(scalar_min, scalar_max, 40)

c_data = ax.contour(X, Y, streamfunction_0.reshape(X.shape), 
                        levels=counter_levels,
                        linewidths=0.5,
                        linestyles='solid',
                        alpha=1.0,
                        cmap=plt.get_cmap("jet"),
                    )


plot_data, = ax.plot(xp,yp,'ob')

# def init():
#     return plot_data

def frame_update(frame_i):

    global xp, yp, plot_data, c_data
    global music_start, annotation1, annotation2

    # strength_i = 500*np.sin(2*np.pi*frame_i/cycle_duration_frames)+1

    strength_i = sample_envelope_dB[frame_i]*10
    # print('strength_i = ', strength_i)

    field_i = Flowfield([
        Vortex(strength_i,5.0,0.0),
        Vortex(-strength_i,-5.0,0.0),
        # Vortex(strength_i,5.0,5.0),
        # Vortex(strength_i,-5.0,-5.0),
        # Vortex(-strength_i,5.0,-5.0),
        # Vortex(-strength_i,-5.0,5.0),
        # Source(strength_i,5.0,0.0),
        # Source(-strength_i,-5.0,0.0),
        # Freestream(0.0,1.0)
    ])

    ###########################################################################

    streamfunction_i = field_i.get_scalar("streamfunction", x_points, y_points)

    # scalar_min = np.nanpercentile(streamfunction_i, 50 - streamfunction_i / 2)
    # scalar_max = np.nanpercentile(streamfunction_i, 50 + streamfunction_i / 2)
    # counter_levels = np.linspace(scalar_min, scalar_max, 80)

    for pti in range(n_pts):

        x_vel = field_i.get_scalar("xvel", xp[pti], yp[pti])
        y_vel = field_i.get_scalar("yvel", xp[pti], yp[pti])

        if not(np.isfinite(x_vel) and np.isfinite(y_vel)):
            print("vel infinite")
            continue
        else:

            xp[pti] += x_vel*dt
            yp[pti] += y_vel*dt

    plot_data.set_data(xp, yp)
    
    # Clear contour data
    for coll in c_data.collections:
        coll.remove()

    # replot frame contours
    c_data = ax.contour(X, Y, streamfunction_i.reshape(X.shape),
                            levels=counter_levels,
                            linewidths=0.5,
                            linestyles='solid',
                            alpha=1.0,
                            cmap=plt.get_cmap("jet"),
                        )

    return plot_data

ani = animation.FuncAnimation(fig, frame_update, frames=sample_num_frames, repeat=False)
                    
# Write video animation file
ani.save(video_output_filename, fps=fps)
plt.show(block=False)

# Write sudio sample file
soundfile.write(audio_output_filename, sample_data, sample_rate, format='wav', subtype='PCM_24')

# Add audio track
subprocess.run(['ffmpeg', '-i', video_output_filename, '-i', audio_output_filename, '-c:v', 'copy', '-c:a', 'aac', av_output_filename, '-strict', '-2'])

# Open rendered video in default system app
subprocess.run(('xdg-open', av_output_filename))

exit()