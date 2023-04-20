from potentialflow import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time



x_points = np.linspace(-10, 10, 200)
y_points = np.linspace(-10, 10, 200)
percentiles_to_include = 99.5


field_0 = Flowfield([
    Vortex(500,5.0,0.0),
    Vortex(-500,-5.0,0.0),
    Source(100,5.0,0.0),
    Source(-100,-5.0,0.0),
    Freestream(0.0,5.0) # Freestream Flow Up Slowly (lavalamp)
])

X,Y,points = field_0.mesh_points(x_points,y_points)

streamfunction_0 = field_0.get_scalar("streamfunction", x_points, y_points)

# field_0.draw("streamfunction", x_points, y_points)

xp_0 = np.array([5.0, -5.0])
yp_0 = np.array([-10.0, -10.0])

xp = xp_0
yp = yp_0

n_pts = xp.size

fps = 100
dt = 1/fps # Delay between frames in seconds.
dt_ms = dt*1000.0; # Delay in milliseconds

########################################################################

fig, ax = plt.subplots()

ax.set_facecolor("black")
ax.grid(False)
# ax.set_axis_off()
ax.margins(x=0)

scalar_min = np.nanpercentile(streamfunction_0, 50 - percentiles_to_include / 2)
scalar_max = np.nanpercentile(streamfunction_0, 50 + percentiles_to_include / 2)
counter_levels = np.linspace(scalar_min, scalar_max, 80)

c_data = ax.contour(X, Y, streamfunction_0.reshape(X.shape), 
                        levels=counter_levels,
                        linelabels=False,
                        linewidths=0.5,
                        linestyles='solid',
                        alpha=1.0,
                        cmap=plt.get_cmap("rainbow"),
                    )


plot_data, = ax.plot(xp,yp,'ob')

# def init():
#     return plot_data

def update(frame):

    global xp, yp, c_data, plot_data

    source_strength_i = 500*np.sin(2*np.pi*frame/100)+1
    
    field_i = Flowfield([
        Vortex(source_strength_i,5.0,0.0),
        Vortex(-source_strength_i,-5.0,0.0),
        Source(source_strength_i,5.0,0.0),
        Source(-source_strength_i,-5.0,0.0),
        Freestream(0.0,1.0)
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
                            linelabels=False,
                            linewidths=0.5,
                            linestyles='solid',
                            alpha=1.0,
                            cmap=plt.get_cmap("jet"),
                        )

    return plot_data,c_data

ani = animation.FuncAnimation(fig, update, frames=np.linspace(0, 100, 100), repeat=True)
                    
plt.show()








# To save the animation, use e.g.
#
# ani.save("movie.mp4")
#
# or
#
# writer = animation.FFMpegWriter(
#     fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)

