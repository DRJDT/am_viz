from potentialflow import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import librosa
# from scipy import signal
# import soundfile 
import time
import subprocess, os

################################################################################

# samples_dir = '/home/jd/Documents/Miscellaneous/ames_music_visualizer/samples/'
# sample_filename = samples_dir + 'Everbloom_2-4-23_Master_2.wav'

sample_filename = librosa.ex('trumpet') # Short sample sound

renders_dir = '/home/jd/Documents/Miscellaneous/ames_music_visualizer/renders/'
render_filename = renders_dir + 'basic_animation.mp4'

fps = 30 # Rendered viualization framerate
dt = 1.0/fps # Delay between frames in seconds.
dt_ms = dt*1000.0; # Delay in milliseconds

cycle_duration_frames = 100

#################################################################################

y, sample_rate = librosa.load(sample_filename, mono=True, offset=0.0, duration=3.0)

num_samples_raw = y.size

sample_duration_sec = sample_duration_num_samples_raw/sample_rate

print('sample_duration_num_samples_raw = ',sample_duration_num_samples_raw)
print('sample_duration_sec = ',sample_duration_sed)

fig, ax = plt.subplots()
librosa.display.waveshow(y, sr=sample_rate, ax=ax)
plt.show()


hop_length = 1024
D = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop_length)),ref=np.max)

fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
img = librosa.display.specshow(D, y_axis='log', sr=sample_rate, x_axis='time', ax=ax[1], hop_length=hop_length)

ax[1].set(title='Log-frequency power spectrogram')
ax[1].label_outer()
fig.colorbar(img, ax=ax, format="%+2.f dB")

plt.show()

exit()

#################################################################################

x_points = np.linspace(-10, 10, 200)
y_points = np.linspace(-10, 10, 100)
percentiles_to_include = 99.5

field_0 = Flowfield([
    # Vortex(500,5.0,0.0),
    # Vortex(-500,-5.0,0.0),
    Vortex(500,5.0,5.0),
    Vortex(500,-5.0,-5.0),
    Vortex(-500,5.0,-5.0),
    Vortex(-500,-5.0,5.0),
    # Source(100,5.0,0.0),
    # Source(-100,-5.0,0.0),
    # Freestream(0.0,5.0) # Freestream Flow Up Slowly (lavalamp)
])

X,Y,points = field_0.mesh_points(x_points,y_points)

streamfunction_0 = field_0.get_scalar("streamfunction", x_points, y_points)

# field_0.draw("streamfunction", x_points, y_points)

# xp_0 = np.array([5.0, -5.0])
# yp_0 = np.array([-10.0, -10.0])

xp_0 = np.array([5.0, -5.0, 0.0, 0.0])
yp_0 = np.array([0.0, 0.0, 5.0, -5.0])

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

def frame_update(frame):

    global xp, yp, plot_data, c_data
    global music_start, annotation1, annotation2

    if frame == 0:
        # music_thread.start()
        music_start = time.perf_counter()

    strength_i = 500*np.sin(2*np.pi*frame/cycle_duration_frames)+1
    
    field_i = Flowfield([
        Vortex(strength_i,5.0,5.0),
        Vortex(strength_i,-5.0,-5.0),
        Vortex(-strength_i,5.0,-5.0),
        Vortex(-strength_i,-5.0,5.0),
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

ani = animation.FuncAnimation(fig, frame_update, frames=cycle_duration_frames, repeat=False)
                    
ani.save(render_filename, fps=fps)

plt.show(block=False)

# Open rendered video in default system app (e.g. VLC)
# subprocess.call(('xdg-open', render_filename))


