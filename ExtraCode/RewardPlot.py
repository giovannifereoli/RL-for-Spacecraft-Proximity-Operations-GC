# Plot for reward explanation
# Libraries
import numpy as np
import matplotlib.pyplot as plt


# Definitions
def R_instance(x, y):
    R = (np.log(x) ** 2 - np.exp(y) ** 2)
    return R


# Initialization
x_vec = np.linspace(0.1, 1, 30)
y_vec = np.linspace(1e-6, 1, 30)
X, Y = np.meshgrid(x_vec, y_vec)

R_vec = R_instance(X, Y)


# Plot
# Creating figure
fig = plt.figure()
ax = plt.axes(projection="3d")
img = ax.plot_surface(X, Y, R_vec, cmap="coolwarm", edgecolor='darkred', linewidth=0.1)
ax.set_xlabel("Bonus [-]")
ax.set_ylabel("Penalty [-]")
ax.set_zlabel("$R^*$ [-]")
ax.view_init(elev=6, azim=-34)
plt.savefig("Reward_logic.pdf")  # Save
plt.show()

