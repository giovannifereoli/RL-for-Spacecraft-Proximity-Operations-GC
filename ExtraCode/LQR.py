import numpy as np
import control as ct
import scipy.optimize
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# Functions
# Functions needed
def rel_crtbp(x):
    """
                Circular Restricted Three-Body Problem Dynamics
    :
                :param x: State, vector 12x1
                :return: State Derivative, vector 6x1
    """

    # Initialize ODE
    dxdt = np.zeros((12,))
    mu = 0.012150583925359
    # Initialize Target State
    xt = x[0]
    yt = x[1]
    zt = x[2]
    xtdot = x[3]
    ytdot = x[4]
    ztdot = x[5]
    # Initialize Relative State
    xr = x[6]
    yr = x[7]
    zr = x[8]
    xrdot = x[9]
    yrdot = x[10]
    zrdot = x[11]

    # Relative CRTBP Dynamics
    r1t = [xt + mu, yt, zt]
    r2t = [xt + mu - 1, yt, zt]
    r1t_norm = np.sqrt((xt + mu) ** 2 + yt**2 + zt**2)
    r2t_norm = np.sqrt((xt + mu - 1) ** 2 + yt**2 + zt**2)
    rho = [xr, yr, zr]

    # Target Equations
    dxdt[0:3] = [xtdot, ytdot, ztdot]
    dxdt[3:6] = [
        2 * ytdot
        + xt
        - (1 - mu) * (xt + mu) / r1t_norm**3
        - mu * (xt + mu - 1) / r2t_norm**3,
        -2 * xtdot + yt - (1 - mu) * yt / r1t_norm**3 - mu * yt / r2t_norm**3,
        -(1 - mu) * zt / r1t_norm**3 - mu * zt / r2t_norm**3,
    ]

    # Chaser equations
    dxdt[6:9] = [xrdot, yrdot, zrdot]
    dxdt[9:12] = [
        2 * yrdot
        + xr
        + (1 - mu)
        * (
            (xt + mu) / r1t_norm**3
            - (xt + xr + mu) / np.linalg.norm(np.add(r1t, rho)) ** 3
        )
        + mu
        * (
            (xt + mu - 1) / r2t_norm**3
            - (xt + xr + mu - 1) / np.linalg.norm(np.add(r2t, rho)) ** 3
        ),
        -2 * xrdot
        + yr
        + (1 - mu)
        * (yt / r1t_norm**3 - (yt + yr) / np.linalg.norm(np.add(r1t, rho)) ** 3)
        + mu * (yt / r2t_norm**3 - (yt + yr) / np.linalg.norm(np.add(r2t, rho)) ** 3),
        (1 - mu)
        * (zt / r1t_norm**3 - (zt + zr) / np.linalg.norm(np.add(r1t, rho)) ** 3)
        + mu * (zt / r2t_norm**3 - (zt + zr) / np.linalg.norm(np.add(r2t, rho)) ** 3)
    ]

    return dxdt


def rel_crtbpT(
        t,
        x,
        K
):
    """
                Circular Restricted Three-Body Problem Dynamics
    :
                :param t: time
                :param x: State, vector 13x1
                :param K: LQR gain
                :return: State Derivative, vector 6x1
    """

    # Initialize ODE
    dxdt = np.zeros((13,))
    m_star = 6.0458 * 1e24  # Kilograms
    l_star = 3.844 * 1e8  # Meters
    t_star = 375200  # Seconds
    mu = 0.012150583925359
    max_thrust = 29620 / (m_star * l_star / t_star ** 2)
    spec_impulse = 310 / t_star
    g0 = 9.81 / (l_star / t_star ** 2)
    # Initialize Target State
    xt = x[0]
    yt = x[1]
    zt = x[2]
    xtdot = x[3]
    ytdot = x[4]
    ztdot = x[5]
    # Initialize Relative State
    xr = x[6]
    yr = x[7]
    zr = x[8]
    xrdot = x[9]
    yrdot = x[10]
    zrdot = x[11]
    # Initial Mass Target
    m = x[12]
    # Initialize Thrust action
    T = - max_thrust * 1e3 * np.dot(K, x[6:12])
    Tx = T[0]
    Ty = T[1]
    Tz = T[2]
    T_norm = np.linalg.norm(T)

    # Relative CRTBP Dynamics
    r1t = [xt + mu, yt, zt]
    r2t = [xt + mu - 1, yt, zt]
    r1t_norm = np.sqrt((xt + mu) ** 2 + yt ** 2 + zt ** 2)
    r2t_norm = np.sqrt((xt + mu - 1) ** 2 + yt ** 2 + zt ** 2) + t * 0
    rho = [xr, yr, zr]

    # Target Equations
    dxdt[0:3] = [xtdot, ytdot, ztdot]
    dxdt[3:6] = [
        2 * ytdot
        + xt
        - (1 - mu) * (xt + mu) / r1t_norm ** 3
        - mu * (xt + mu - 1) / r2t_norm ** 3,
        -2 * xtdot
        + yt
        - (1 - mu) * yt / r1t_norm ** 3
        - mu * yt / r2t_norm ** 3,
        -(1 - mu) * zt / r1t_norm ** 3 - mu * zt / r2t_norm ** 3,
    ]

    # Chaser equations
    dxdt[6:9] = [xrdot, yrdot, zrdot]
    dxdt[9:12] = [
        2 * yrdot
        + xr
        + (1 - mu)
        * (
                (xt + mu) / r1t_norm ** 3
                - (xt + xr + mu) / np.linalg.norm(np.add(r1t, rho)) ** 3
        )
        + mu
        * (
                (xt + mu - 1) / r2t_norm ** 3
                - (xt + xr + mu - 1) / np.linalg.norm(np.add(r2t, rho)) ** 3
        )
        + Tx / m,
        -2 * xrdot
        + yr
        + (1 - mu)
        * (
                yt / r1t_norm ** 3
                - (yt + yr) / np.linalg.norm(np.add(r1t, rho)) ** 3
        )
        + mu
        * (
                yt / r2t_norm ** 3
                - (yt + yr) / np.linalg.norm(np.add(r2t, rho)) ** 3
        )
        + Ty / m,
        (1 - mu)
        * (
                zt / r1t_norm ** 3
                - (zt + zr) / np.linalg.norm(np.add(r1t, rho)) ** 3
        )
        + mu
        * (
                zt / r2t_norm ** 3
                - (zt + zr) / np.linalg.norm(np.add(r2t, rho)) ** 3
        )
        + Tz / m
    ]
    dxdt[12] = - T_norm / (spec_impulse * g0)

    return dxdt


# Data
m_star = 6.0458 * 1e24  # Kilograms
l_star = 3.844 * 1e8  # Meters
t_star = 375200  # Seconds
mass = 21000
max_thrust = 29620

# Initial conditions (1000m)
x0_target = np.array(
    [
        1.02206694e00,
        -1.32282592e-07,
        -1.82100000e-01,
        -1.69229909e-07,
        -1.03353155e-01,
        6.44013821e-07,
    ]
)  # 9:2 NRO - 50m after apolune, already corrected, rt = 399069639.7170633, vt = 105.88740083894766
x0_relative = np.array(
    [
        1.08357767e-13,
        1.32282592e-07,
        -4.12142542e-13,
        1.69229909e-07,
        -3.65860120e-13,
        -6.44013821e-07,
    ]
)
x0_mass = np.array([mass / m_star])
x0 = np.concatenate((x0_target, x0_relative))
x02 = np.concatenate((x0_target, x0_relative, x0_mass))

# Matrices
A = scipy.optimize.approx_fprime(f=rel_crtbp, xk=x0)
A = A[6:12, 6:12]
B = np.array(([0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]))
Q = np.eye(6)  # 1e2 * np.eye(6)
R = np.eye(3)  # (max_thrust ** - 2) * np.eye(3)   # OSS: Bryson's rule

# LQR
K, S, E = ct.lqr(A, B, Q, R)
print(K)
print(S)
print(E)

# Integration
ToF = 10 * (np.pi / 2)   # OSS: sono sette orbite, 65 giorni
disc = 1000
dt = ToF / 1000
sol = solve_ivp(
    fun=rel_crtbpT,
    t_span=(0, ToF),
    t_eval=np.linspace(0, ToF, disc),
    y0=x02,
    method="LSODA",
    rtol=2.220446049250313e-14,
    atol=2.220446049250313e-14,
    args=(K, )  # OSS: it shall be a tuple
)
state = np.transpose(sol.y)
print(state)

# Approach Corridor
rad_kso = 50
ang_corr = np.deg2rad(15)
rad_entry = np.tan(ang_corr) * rad_kso
x_cone, y_cone = np.mgrid[-rad_entry:rad_entry:1000j, -rad_entry:rad_entry:1000j]
x_cone = x_cone
y_cone = y_cone
z_cone = np.sqrt((x_cone**2 + y_cone**2) / np.square(np.tan(ang_corr)))
z_cone = np.where(z_cone > rad_kso, np.nan, z_cone)

# Plot Chaser Relative
xr_sol = state[:, 6:9] * l_star
x_mass = state[:, -1] * m_star
plt.figure(1)
ax = plt.axes(projection="3d")
ax.plot3D(xr_sol[:, 0], xr_sol[:, 1], xr_sol[:, 2], "b", linewidth=2)
ax.plot3D(0, 0, 0, "ko", markersize=5)
ax.plot3D(xr_sol[0, 0], xr_sol[0, 1], xr_sol[0, 2], "go", markersize=5)
ax.plot3D(xr_sol[-1, 0], xr_sol[-1, 1], xr_sol[-1, 2], "ro", markersize=5)
ax.legend(["Trajectory", "Target", "Initial State", "Final State"], ncol=2, loc="lower center")
ax.plot_surface(x_cone, z_cone, y_cone, color="k", alpha=0.1)
ax.set_xlabel("$\delta x$ [DU]")
ax.set_ylabel("$\delta y$ [DU]")
ax.zaxis.set_rotate_label(False)
ax.set_zlabel("$\delta z$ [DU]", rotation=-45, labelpad=5)
ax.set_zticks([0])
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
ax.view_init(elev=90, azim=0)
plt.savefig(".\LQR_Trajectory.pdf")  # Save
plt.show()

# Plot mass usage
plt.close()  # Initialize
plt.figure(2)
plt.ticklabel_format(scilimits=(-5, 8))
plt.ticklabel_format(style='plain', useOffset=False, axis='y')
plt.plot(np.linspace(0, len(x_mass), len(x_mass)), x_mass, c="r", linewidth=2)
plt.grid(True)
plt.xlabel("Step [-]")
plt.ylabel("Mass [kg]")  # OSS: 1ish kg
plt.show()

# Thrust plot
x = state[:, 6:12]
Tx = np.zeros(len(x_mass))
Ty = np.zeros(len(x_mass))
Tz = np.zeros(len(x_mass))
for i in range(len(x_mass)):
    Thrust = - max_thrust * np.dot(K, x[i, :])
    Tx[i] = Thrust[0]
    Ty[i] = Thrust[1]
    Tz[i] = Thrust[2]
plt.close()  # Initialize
plt.figure(3)
plt.semilogy(np.linspace(0, len(x_mass), len(x_mass)), Tx, c="r", linewidth=2, label='Tx')
plt.semilogy(np.linspace(0, len(x_mass), len(x_mass)), Ty, c="b", linewidth=2, label='Ty')
plt.semilogy(np.linspace(0, len(x_mass), len(x_mass)), Tz, c="g", linewidth=2, label='Tz')
plt.legend()
plt.grid(True)
plt.xlabel("Step [-]")
plt.ylabel("Thrust [N]")
plt.show()

# Plot angular velocity
dTdt_ver = np.zeros([len(x_mass), 3])
w_ang = np.zeros(len(x_mass))
w_ang[0] = np.nan
thrust = np.transpose(np.vstack([Tx, Ty, Tz]))
dTdt_ver = np.diff(thrust / np.linalg.norm(thrust), axis=0) / dt   # Finite difference
Tb_ver = np.array([1, 0, 0])
for i in range(len(w_ang) - 1):  # OSS: T aligned with x-axis body-frame assumptions.
    wy = dTdt_ver[i, 2] / Tb_ver[0]
    wz = - dTdt_ver[i, 1] / Tb_ver[0]
    w_ang[i + 1] = np.rad2deg(np.linalg.norm(np.array([0, wy, wz])))
plt.close()  # Initialize
plt.figure()
plt.semilogy(np.linspace(0, len(x_mass), len(x_mass)), w_ang, c="c", linewidth=2)
plt.grid(True)
plt.xlabel("Step [-]")
plt.ylabel("Angular velocity [deg/s]")
plt.show()


