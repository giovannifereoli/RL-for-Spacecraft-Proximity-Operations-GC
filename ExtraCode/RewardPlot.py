# Plot for reward explanation
# Libraries
import numpy as np
import matplotlib.pyplot as plt


# Definitions
def R_instance(x, y, z):
    R = np.log(x) ** 2 - np.exp(y) ** 2 - np.exp(z) ** 2
    return R


# Initialization
x_vec = np.linspace(1e-6, 1, 30)
y_vec = np.linspace(1e-6, 1, 30)
z_vec = np.linspace(1e-6, 1, 30)
X, Y, Z = np.meshgrid(x_vec, y_vec, z_vec)

R_vec = R_instance(X, Y, Z)


# Plot
# Creating figure
fig = plt.figure()
ax = plt.axes(projection="3d")
img = ax.scatter3D(X, Y, Z, c= R_vec / 20, alpha=0.7, marker='.')
ax.set_xlabel("$pos$ [-]")
ax.set_ylabel("$ang$ [-]")
ax.set_zlabel("$T$ [-]")
fig.colorbar(img)
plt.show()
