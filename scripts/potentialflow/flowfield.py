import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.tools.pretty_plots import plt, show_plot, contour, mpl

class Flowfield:
    def __init__(self,
                 objects=[]
                 ):
        self.objects = objects

    def mesh_points(self,
            x_points: np.ndarray=np.linspace(-10, 10, 100),
            y_points: np.ndarray=np.linspace(-10, 10, 100),
            ):
        
        X, Y = np.meshgrid(x_points, y_points)
        X_r = np.reshape(X, -1)
        Y_r = np.reshape(Y, -1)
        points = np.vstack((X_r, Y_r)).T

        return X,Y,points

    def get_scalar(self,
             scalar_to_plot:str="potential",  # "potential", "streamfunction", "xvel", "yvel", "velmag", "Cp"
             x_points: np.ndarray=np.linspace(-10, 10, 100),
             y_points: np.ndarray=np.linspace(-10, 10, 100)
             ):
        
        X,Y,points = self.mesh_points(x_points,y_points)

        if scalar_to_plot == "potential":
            scalar_to_plot_value = sum([object.get_potential_at(points) for object in self.objects])
        elif scalar_to_plot == "streamfunction":
            scalar_to_plot_value = sum([object.get_streamfunction_at(points) for object in self.objects])
        elif scalar_to_plot == "xvel":
            scalar_to_plot_value = sum([object.get_x_velocity_at(points) for object in self.objects])
        elif scalar_to_plot == "yvel":
            scalar_to_plot_value= sum([object.get_y_velocity_at(points) for object in self.objects])
        elif scalar_to_plot == "velmag":
            x_vels = sum([object.get_x_velocity_at(points) for object in self.objects])
            y_vels = sum([object.get_y_velocity_at(points) for object in self.objects])
            scalar_to_plot_value = np.sqrt(x_vels ** 2 + y_vels ** 2)
        elif scalar_to_plot == "Cp":
            x_vels = sum([object.get_x_velocity_at(points) for object in self.objects])
            y_vels = sum([object.get_y_velocity_at(points) for object in self.objects])
            V = np.sqrt(x_vels ** 2 + y_vels ** 2)
            scalar_to_plot_value = 1 - V ** 2
        else:
            raise ValueError("Bad value of `scalar_to_plot`!")

        return scalar_to_plot_value
    
    def draw(self,
             scalar_to_plot:str="potential",  # "potential", "streamfunction", "xvel", "yvel", "velmag", "Cp"
             x_points: np.ndarray=np.linspace(-10, 10, 100),
             y_points: np.ndarray=np.linspace(-10, 10, 100),
             percentiles_to_include = 99.7,
             show=True,
             ):

        X,Y,points = self.mesh_points(x_points,y_points)

        scalar_to_plot_value = self.get_scalar(scalar_to_plot, x_points, y_points)

        scalar_min = np.nanpercentile(scalar_to_plot_value, 50 - percentiles_to_include / 2)
        scalar_max = np.nanpercentile(scalar_to_plot_value, 50 + percentiles_to_include / 2)

        contour(
            x_points, y_points, scalar_to_plot_value.reshape(X.shape),
            levels=np.linspace(scalar_min, scalar_max, 80),
            linelabels=False,
            cmap=plt.get_cmap("rainbow"),
            contour_kwargs={
                "linestyles": 'solid',
                "alpha": 0.4
            }
        )
        plt.axis("equal")
        show_plot(
            scalar_to_plot,
            "$x$",
            "$y$",
            show=show
        )