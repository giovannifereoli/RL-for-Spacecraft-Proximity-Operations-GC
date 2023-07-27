from mpopt import mp
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt


def dynamics(x, u, t):
    # Initialize ODE
    mu = 0.012150583925359
    m_star = 6.0458 * 1e24  # Kilograms
    l_star = 3.844 * 1e8  # Meters
    t_star = 375200  # Seconds
    Isp = 310.0 / t_star
    g0 = 9.81 / (l_star / t_star**2)
    Tmax = 29620 / (m_star * l_star / t_star**2)
    # Initialize Target State
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
    m = x[12]

    # Relative CRTBP Dynamics
    r1t = [xt + mu, yt, zt]
    r2t = [xt + mu - 1, yt, zt]
    r1t_norm = ca.sqrt((xt + mu) ** 2 + yt**2 + zt**2)
    r2t_norm = ca.sqrt((xt + mu - 1) ** 2 + yt**2 + zt**2)
    rho = [xr, yr, zr]
    r1t_rho = np.add(r1t, rho)
    r1t_rho_norm = ca.sqrt(r1t_rho[0] ** 2 + r1t_rho[1] ** 2 + r1t_rho[2] ** 2)
    r2t_rho = np.add(r2t, rho)
    r2t_rho_norm = ca.sqrt(r2t_rho[0] ** 2 + r2t_rho[1] ** 2 + r2t_rho[2] ** 2)

    return [
        xtdot,
        ytdot,
        ztdot,
        2 * ytdot
        + xt
        - (1 - mu) * (xt + mu) / r1t_norm**3
        - mu * (xt + mu - 1) / r2t_norm**3,
        -2 * xtdot + yt - (1 - mu) * yt / r1t_norm**3 - mu * yt / r2t_norm**3,
        -(1 - mu) * zt / r1t_norm**3 - mu * zt / r2t_norm**3,
        xrdot,
        yrdot,
        zrdot,
        2 * yrdot
        + xr
        + (1 - mu) * ((xt + mu) / r1t_norm**3 - (xt + xr + mu) / r1t_rho_norm**3)
        + mu * ((xt + mu - 1) / r2t_norm**3 - (xt + xr + mu - 1) / r2t_rho_norm**3)
        + Tmax * u[0] * u[3] / (m * ca.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)),
        -2 * xrdot
        + yr
        + (1 - mu) * (yt / r1t_norm**3 - (yt + yr) / r1t_rho_norm**3)
        + mu * (yt / r2t_norm**3 - (yt + yr) / r2t_rho_norm**3)
        + Tmax * u[1] * u[3] / (m * ca.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)),
        (1 - mu) * (zt / r1t_norm**3 - (zt + zr) / r1t_rho_norm**3)
        + mu * (zt / r2t_norm**3 - (zt + zr) / r2t_rho_norm**3)
        + Tmax * u[2] * u[3] / (m * ca.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)),
        -Tmax * u[3] / (Isp * g0),
    ]


def running_costs(x, u, t):
    u_vec = u[3] * np.array([u[0], u[1], u[2]])
    return u_vec[0] * u_vec[0] + u_vec[1] * u_vec[1] + u_vec[2] * u_vec[2]


def path_constraints(x, u, t):
    return [
        x[7] ** 2 - (x[6] ** 2 + x[7] ** 2 + x[8] ** 2) * ca.cos(np.deg2rad(20)) ** 2
    ]


def terminal_constraints(x, t, x0, t0):
    return [x[6], x[7], x[8], x[9], x[10], x[11]]


ocp = mp.OCP(
    n_states=13,
    n_controls=4,
    n_phases=1,
    nlp_solver_options={"ipopt.acceptable_tol": 1e-6, "ipopt.max_iter": 100000}
)
ocp.dynamics[0] = dynamics
ocp.running_costs[0] = running_costs
ocp.terminal_constraints[0] = terminal_constraints
ocp.path_constraints[0] = path_constraints

# Data
m_star = 6.0458 * 1e24  # Kilograms
l_star = 3.844 * 1e8  # Meters
t_star = 375200  # Seconds
mass = 21000
max_thrust = 29620

# Initial state
ocp.x00[0] = [
    1.02206694e00,
    -1.32282592e-07,
    -1.82100000e-01,
    -1.69229909e-07,
    -1.03353155e-01,
    6.44013821e-07,
    1.08357767e-13,
    1.32282592e-07,
    -4.12142542e-13,
    1.69229909e-07,
    -3.65860120e-13,
    -6.44013821e-07,
    mass / m_star,
]
ocp.xf0[0] = [
    1.02206694e00,
    -1.32282592e-07,
    -1.82100000e-01,
    -1.69229909e-07,
    -1.03353155e-01,
    6.44013821e-07,
    0,
    0,
    0,
    0,
    0,
    0,
    0.8 * mass / m_star,
]
ocp.u00[0], ocp.uf0[0] = [0, -1, 0, 1], [0, 0, 0, 0]
ocp.t00[0] = 0

# Box constraints
ocp.lbx[0] = [
    379548434.40513575 / l_star,
    -16223383.008425826 / l_star,
    -70002940.10058032 / l_star,
    -81.99561388926969 / (l_star / t_star),
    -105.88740121359594 / (l_star / t_star),
    -881.9954974936014 / (l_star / t_star),
    -55 / l_star,
    -55 / l_star,
    -55 / l_star,
    -5 / (l_star / t_star),
    -5 / (l_star / t_star),
    -5 / (l_star / t_star),
    0.6 * mass / m_star,
]
ocp.ubx[0] = [
    392882530.7281463 / l_star,
    16218212.912172267 / l_star,
    3248770.078052207 / l_star,
    82.13051133777446 / (l_star / t_star),
    1707.5720010497114 / (l_star / t_star),
    881.8822374702228 / (l_star / t_star),
    55 / l_star,
    55 / l_star,
    55 / l_star,
    5 / (l_star / t_star),
    5 / (l_star / t_star),
    5 / (l_star / t_star),
    mass / m_star,
]
ocp.lbu[0], ocp.ubu[0] = [-1, -1, -1, 0], [1, 1, 1, 1]
ocp.lbtf[0], ocp.ubtf[0] = 20 / t_star, 40 / t_star
ocp.lbt0[0], ocp.ubt0[0] = 0, 0

ocp.scale_x = [
    l_star / 392882530,
    l_star / 392882530,
    l_star / 392882530,
    l_star / t_star / 1707,
    l_star / t_star / 1707,
    l_star / t_star / 1707,
    l_star / 50,
    l_star / 50,
    l_star / 50,
    l_star / t_star / 0.5,
    l_star / t_star / 0.5,
    l_star / t_star / 0.5,
    m_star / 21000
]
ocp.scale_t = t_star / 30

ocp.validate()
mpo, post = mp.solve(ocp, n_segments=1, poly_orders=4, scheme="LGR", plot=True)

mp.post_process._INTERPOLATION_NODES_PER_SEG = 200
x0, u0, t0, _ = post.get_data(phases=0, interpolate=True)
mp.plt.show()
t0 = t0 * t_star
rf = np.sqrt(x0[-1, 6] ** 2 + x0[-1, 7] ** 2 + x0[-1, 8] ** 2) * l_star
vf = np.sqrt(x0[-1, 9] ** 2 + x0[-1, 10] ** 2 + x0[-1, 11] ** 2) * l_star / t_star


# Approach Corridor
rad_kso = 50
ang_corr = np.deg2rad(20)
rad_entry = np.tan(ang_corr) * rad_kso
x_cone, y_cone = np.mgrid[-rad_entry:rad_entry:1000j, -rad_entry:rad_entry:1000j]
z_cone = np.sqrt((x_cone**2 + y_cone**2) / np.square(np.tan(ang_corr)))
z_cone = np.where(z_cone > rad_kso, np.nan, z_cone)

# Plot Chaser Relative
xr_sol = x0[:, 6:9] * l_star
x_mass = x0[:, -1]
dV = 310 * 9.81 * np.log(x_mass[0] / x_mass[-1])
plt.figure(1)
ax = plt.axes(projection="3d")
ax.plot3D(xr_sol[:, 0], xr_sol[:, 1], xr_sol[:, 2], "b", linewidth=2)
ax.plot3D(0, 0, 0, "ko", markersize=5)
ax.plot3D(xr_sol[0, 0], xr_sol[0, 1], xr_sol[0, 2], "go", markersize=5)
ax.plot3D(xr_sol[-1, 0], xr_sol[-1, 1], xr_sol[-1, 2], "ro", markersize=5)
ax.legend(
    ["Trajectory", "Target", "Initial State", "Final State"], ncol=2, loc="lower center"
)
ax.plot_surface(x_cone, z_cone, y_cone, color="k", alpha=0.1)
ax.set_xlabel("$\delta x$ [m]")
ax.set_ylabel("$\delta y$ [m]")
ax.zaxis.set_rotate_label(False)
ax.set_zlabel("$\delta z$ [m]", rotation=-45, labelpad=5)
ax.set_zticks([0])
plt.locator_params(axis="x", nbins=4)
plt.locator_params(axis="y", nbins=4)
plt.locator_params(axis="z", nbins=4)
# ax.set_aspect("equal", "box")
ax.xaxis.pane.set_edgecolor("black")
ax.yaxis.pane.set_edgecolor("black")
ax.zaxis.pane.set_edgecolor("black")
plt.tick_params(axis="z", which="major", pad=10)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.set_title(
    "$r_f$, $v_f$: [%.3f m, %.3f m/s]  - $\Delta V$: %.3f m/s "
    % (rf, vf, dV),
    y=1,
    pad=30,
)
ax.view_init(elev=90, azim=0)
plt.savefig(".\OCP.pdf")

# Thrust plot
T = 29620
x = u0
Tx = T * np.multiply(u0[:, 0], u0[:, 3])
Ty = T * np.multiply(u0[:, 1], u0[:, 3])
Tz = T * np.multiply(u0[:, 2], u0[:, 3])
plt.close()  # Initialize
plt.figure(3)
plt.plot(t0, Tx, c="r", linewidth=2, label="Tx")
plt.plot(t0, Ty, c="b", linewidth=2, label="Ty")
plt.plot(t0, Tz, c="g", linewidth=2, label="Tz")
plt.legend()
plt.grid(True)
plt.xlabel("Time [s]")
plt.ylabel("Thrust [N]")
plt.show()


