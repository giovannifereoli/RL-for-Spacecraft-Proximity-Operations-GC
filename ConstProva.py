# Import libraries
import numpy as np
import matplotlib.pyplot as plt

# Data
mu = 0.012150583925359
m_star = 6.0458 * 1e24  # Kilograms
l_star = 3.844 * 1e8  # Meters
t_star = 375200  # Seconds

# Check constraints
def is_inside(state):
    # Data and initialization
    l_star = 3.844 * 1e8  # Meters
    xr = state[0] * l_star
    yr = state[1] * l_star
    zr = state[2] * l_star
    ang_corr = np.deg2rad(10)
    rad_kso = 200
    collision = False

    # Computations
    radius_zcone = zr * np.tan(ang_corr)  # Radius of cone at z
    dist_z = np.sqrt(xr ** 2 + yr ** 2)    # Distance from z-axis
    rho = np.sqrt(xr ** 2 + yr ** 2 + zr ** 2)  # Distance from origin

    # Constraints check
    if yr >= 0:
        if dist_z > radius_zcone and rho < rad_kso:  # OSS: if outside cone and inside sphere there's collision!
            collision = True
    else:
        if rho < rad_kso:
            collision = True

    return collision

# Approach Corridor
rad_kso = 200
ang_corr = np.deg2rad(10)
rad_entry = np.tan(ang_corr) * rad_kso
x_cone, y_cone = np.mgrid[-rad_entry:rad_entry:1000j, -rad_entry:rad_entry:1000j]
x_cone = x_cone / l_star
y_cone = y_cone / l_star
z_cone = np.sqrt((x_cone**2 + y_cone**2) / np.square(np.tan(ang_corr)))
z_cone = np.where(z_cone > (rad_kso / l_star), np.nan, z_cone)

# Keep-Out Sphere
x_sph, y_sph = np.mgrid[-rad_kso:rad_kso:1000j, -rad_kso:rad_kso:1000j]
x_sph = x_sph / l_star
y_sph = y_sph / l_star
z_sph1_sq = (rad_kso / l_star) ** 2 - x_sph**2 - y_sph**2
z_sph1_sq = np.where(z_sph1_sq < 0, np.nan, z_sph1_sq)
z_sph1 = np.sqrt(z_sph1_sq)
z_sph2 = - z_sph1

# PROVA
xstate = np.array([30, 150, 20]) / l_star
print(is_inside(xstate))

# Plot Chaser Relative
plt.figure(2)
ax = plt.axes(projection="3d")
ax.plot_surface(x_cone, z_cone, y_cone, color="r")
ax.plot_surface(x_sph, y_sph, z_sph1, color="y", alpha=0.2)
ax.plot_surface(x_sph, y_sph, z_sph2, color="y", alpha=0.2)
ax.set_xlabel("$\delta x$ [DU]")
ax.set_ylabel("$\delta y$ [DU]")
ax.zaxis.set_rotate_label(False)
ax.set_zlabel("$\delta z$ [DU]", rotation=90, labelpad=20)
ax.set_xticks([0])
plt.locator_params(axis="x", nbins=4)
plt.locator_params(axis="y", nbins=4)
plt.locator_params(axis="z", nbins=4)
ax.set_aspect("equal", "box")
ax.xaxis.pane.set_edgecolor("black")
ax.yaxis.pane.set_edgecolor("black")
ax.zaxis.pane.set_edgecolor("black")
plt.tick_params(axis="z", which="major", pad=10)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.view_init(elev=10, azim=30)
plt.show()
