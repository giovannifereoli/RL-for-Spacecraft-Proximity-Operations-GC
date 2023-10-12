# Import libraries
import numpy as np
from sb3_contrib import RecurrentPPO
from EnvironmentPert import ArpodCrtbp
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt
import time

# DEFINITIONS
# Data and initialization
m_star = 6.0458 * 1e24  # Kilograms
l_star = 3.844 * 1e8  # Meters
t_star = 375200  # Seconds

dt = 0.5
ToF = 200
batch_size = 64

rho_max = 70
rhodot_max = 6

ang_corr = np.deg2rad(20)
safety_radius = 1
safety_vel = 0.1

max_thrust = 29620
mass = 21000
Isp = 300
g0 = 9.81
state_space = 16
actions_space = 3

x0t_state = np.array(
    [
        1.02206694e00,
        -1.32282592e-07,
        -1.82100000e-01,
        -1.69229909e-07,
        -1.03353155e-01,
        6.44013821e-07
    ]
)  # 9:2 NRO - 50m after apolune, already corrected, rt = 399069639.7170633, vt = 105.88740083894766
x0r_state = np.array(
    [
        1.08357767e-13,
        1.32282592e-07,
        -4.12142542e-13,
        1.69229909e-07,
        -3.65860120e-13,
        -6.44013821e-07
    ]
)
x0r_mass = np.array([mass / m_star])
x0_time_rem = np.array([ToF / t_star])
x0ivp_vec = np.concatenate((x0t_state, x0r_state, x0r_mass, x0_time_rem))
x0ivp_std_vec = np.absolute(
    np.concatenate(
        (
            np.zeros(6),
            5 * np.ones(3) / l_star,
            0.1 * np.ones(3) / (l_star / t_star),
            0.005 * x0r_mass,
            np.zeros(1)
        )
    )
)

# Define environment and model
env = ArpodCrtbp(
    max_time=ToF,
    dt=dt,
    rho_max=rho_max,
    rhodot_max=rhodot_max,
    x0ivp=x0ivp_vec,
    x0ivp_std=x0ivp_std_vec,
    ang_corr=ang_corr,
    safety_radius=safety_radius,
    safety_vel=safety_vel
)
check_env(env)

# TESTING with MCM
# Loading model and reset environment
model = RecurrentPPO.load("ppo_recurrentBest")
print(model.policy)

# Trajectory propagation
num_episode_MCM = 500
num_ep = 0
docked = np.zeros(num_episode_MCM)
posfin_mean, posfin_std = 0, 0
velfin_mean, velfin_std = 0, 0
dv_mean, dv_std = 0, 0
ToF_mean, ToF_std = 0, 0
tc_mean, tc_std = 0, 0

# Approach Corridor: truncated cone + cylinder
len_cut = np.sqrt((safety_radius**2) / np.square(np.tan(ang_corr)))
rad_kso = rho_max + len_cut
rad_entry = np.tan(ang_corr) * rad_kso
x_cone, z_cone = np.mgrid[-rad_entry:rad_entry:1000j, -rad_entry:rad_entry:1000j]
y_cone = np.sqrt((x_cone**2 + z_cone**2) / np.square(np.tan(ang_corr))) - len_cut
y_cone = np.where(y_cone > rho_max, np.nan, y_cone)
y_cone = np.where(y_cone < 0, np.nan, y_cone)

# Plot
plt.close()
plt.figure()
ax = plt.axes(projection="3d")

# Propagation
for num_ep in range(num_episode_MCM):
    # Initialization
    obs = env.reset()
    obs_vec = env.scaler_reverse_observation(obs)
    lstm_states = None
    done = True

    # Propagation
    while True:
        # Action sampling and propagation
        t1 = time.perf_counter()
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=np.array([done]), deterministic=True
        )  # OSS: Episode start signals are used to reset the lstm states
        obs, rewards, done, info = env.step(action)
        tc = time.perf_counter() - t1

        # Saving
        obs_vec = np.vstack((obs_vec, env.scaler_reverse_observation(obs)))

        # Stop
        if done:
            break

    # Plot
    traj = ax.plot3D(
        obs_vec[:, 6] * l_star,
        obs_vec[:, 7] * l_star,
        obs_vec[:, 8] * l_star,
        c=np.random.rand(
            3,
        ),
        linewidth=1.5,
    )

    # DV and ToF Computation
    dv = Isp * g0 * np.log(obs_vec[0, 12] / obs_vec[-1, 12])
    ToF = len(obs_vec) * dt

    # Check RVD (OSS: it happens at the end)
    if info.get("Episode success") == "docked":
        docked[num_ep] = 1

    # Statistics
    if num_ep == 0:
        posfin_mean = np.linalg.norm(obs_vec[-1, 6:9])
        velfin_mean = np.linalg.norm(obs_vec[-1, 9:12])
        dv_mean = dv
        tc_mean = tc
        ToF_mean = ToF
        posfin_std, velfin_std, dv_std, tc_std, ToF_std = 0, 0, 0, 0, 0
    else:
        posfin_mean = np.mean([posfin_mean, np.linalg.norm(obs_vec[-1, 6:9])])
        velfin_mean = np.mean([velfin_mean, np.linalg.norm(obs_vec[-1, 9:12])])
        dv_mean = np.mean([dv_mean, dv])
        tc_mean = np.mean([tc_mean, tc])
        ToF_mean = np.mean([ToF_mean, ToF])
        posfin_std = np.std(
            [
                posfin_mean + posfin_std,
                posfin_mean - posfin_std,
                np.linalg.norm(obs_vec[-1, 6:9]),
            ]
        )
        velfin_std = np.std(
            [
                velfin_mean + velfin_std,
                velfin_mean - velfin_std,
                np.linalg.norm(obs_vec[-1, 9:12]),
            ]
        )
        dv_std = np.std(
            [
                dv_mean + dv_std,
                dv_mean - dv_std,
                dv,
            ]
        )
        tc_std = np.std(
            [
                tc_mean + tc_std,
                tc_mean - tc_std,
                tc,
            ]
        )
        ToF_std = np.std(
            [
                ToF_mean + ToF_std,
                ToF_mean - ToF_std,
                ToF,
            ]
        )

# Re-scaling and other Statistics
posfin_mean = posfin_mean * l_star
posfin_std = posfin_std * l_star
velfin_mean = velfin_mean * l_star / t_star
velfin_std = velfin_std * l_star / t_star
prob_RVD = docked.sum() * 100 / num_episode_MCM

# Print Info
print("ToF mean and standard deviation:", ToF_mean, ",", ToF_std)
print("Computational cost mean and standard deviation:", tc_mean, ",", tc_std)

# Plot full trajectory statistics
goal = ax.scatter(0, 0, 0, color="red", marker="^", label="Target")
app_direction = ax.plot3D(
    np.zeros(100),
    np.linspace(0, rho_max, 100),
    np.zeros(100),
    color="black",
    linestyle="dashed",
    label="Corridor",
)
ax.set_xlabel("$\delta x$ [m]", labelpad=15)
plt.xticks([0])
ax.plot_surface(x_cone, y_cone, z_cone, color="k", alpha=0.1)
plt.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.88))
ax.set_ylabel("$\delta y$ [m]", labelpad=10)
ax.zaxis.set_rotate_label(False)
ax.set_zlabel("$\delta z$ [m]", labelpad=10, rotation=90)
plt.locator_params(axis="y", nbins=6)
plt.locator_params(axis="z", nbins=6)
ax.xaxis.pane.set_edgecolor("black")
ax.yaxis.pane.set_edgecolor("black")
ax.zaxis.pane.set_edgecolor("black")
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.view_init(elev=0, azim=0)
'''
ax.set_title(
    " $S_r$ : %.1f %% "
    "\n $\mu_{|\mathbf{x}_f|}$: [%.3f m, %.3f m/s] "
    "\n $\sigma_{|\mathbf{x}_f|}$: [%.3f m, %.3f m/s] "
    "\n $\mu_{\Delta V}, \sigma_{\Delta V}$: %.3f m/s, %.3f m/s"
    % (prob_RVD, posfin_mean, velfin_mean, posfin_std, velfin_std, dv_mean, dv_std),
    y=1,
    pad=-3,
)'''
plt.savefig("plots\MCM_TrajectoryPert.pdf")  # Save
print(dv_mean)
plt.show()
