import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Ellipse

def draw_2d(ax, covariance, mean):
    E, V = np.linalg.eig(covariance)
    angle = np.arctan2(V[0,0], V[0,1]) / np.pi * 180.0
    el = Ellipse(xy=mean, width=2*np.sqrt(E[1]*5.99), height=2*np.sqrt(E[0]*5.99), angle=angle,
                 color='C1', alpha=0.5)
    ax.add_artist(el)


def draw_3d(ax, covariance, mean):

    # find the rotation matrix and radii of the axes
    E, V = np.linalg.eig(covariance)
    radii = np.sqrt(E*7.81)  #reduce radii by factor 0.3 

    # calculate cartesian coordinates for the ellipsoid surface
    u = np.linspace(0.0, 2.0 * np.pi, 10)
    v = np.linspace(0.0, np.pi, 6)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    pts = np.stack([x,y,z], axis=-1)
    pts = np.dot(pts, V.transpose()) + mean

    ax.plot_surface(pts[:,:,0], pts[:,:,1], pts[:,:,2], rstride=1, cstride=1,
                    color='C1', linewidth=0.1, alpha=0.5, shade=True)

if __name__ == "__main__":
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    covariance = np.array([[9.0,-4.0,2.0],
                           [-4.0,4.0,0],
                           [2.0,0,1.0]])
    center = np.zeros(3)
    draw_3d(ax, covariance, center)
    plt.show()
