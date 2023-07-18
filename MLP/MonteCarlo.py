# Import libraries
import numpy as np
from stable_baselines3 import PPO
from Environment import ArpodCrtbp
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt

# DEFINITIONS
# Data and initialization
m_star = 6.0458 * 1e24  # Kilograms
l_star = 3.844 * 1e8  # Meters
t_star = 375200  # Seconds

dt = 0.5
ToF = 200
batch_size = 64

rho_max = 60
rhodot_max = 6

ang_corr = np.deg2rad(15)
safety_radius = 1
safety_vel = 0.1

max_thrust = 29620
mass = 21000
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
            2.5 * np.ones(3) / l_star,
            0.5 * np.ones(3) / (l_star / t_star),
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
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    batch_size=batch_size,
    n_steps=int(batch_size * ToF / dt),
    n_epochs=10,
    learning_rate=0.0001,
    gamma=0.99,
    gae_lambda=1,
    clip_range=0.1,
    max_grad_norm=0.1,
    ent_coef=1e-4,
    tensorboard_log="./tensorboard/",
)
print(model.policy)

# TESTING with MCM
# Remove to demonstrate saving and loading
del model

# Loading model and reset environment
model = PPO.load("ppo_mlp")

# Trajectory propagation
num_episode_MCM = 200
num_ep = 0
docked = np.zeros(num_episode_MCM)
obs_mean = np.array([])
obs_std = np.array([])

for num_ep in range(num_episode_MCM):
    # Initialization
    obs = env.reset()
    obs_vec = env.scaler_reverse_observation(obs)
    lstm_states = None
    done = True

    # Propagation
    while True:
        # Action sampling and propagation
        action, _states = model.predict(
            obs, deterministic=True
        )  # OSS: Episode start signals are used to reset the lstm states
        obs, rewards, done, info = env.step(action)

        # Saving
        obs_vec = np.vstack((obs_vec, env.scaler_reverse_observation(obs)))

        # Stop
        if done:
            break

    # Check RVD (OSS: it happens at the end)
    if info.get("Episode success") == "docked":
        docked[num_ep] = 1

    # Statistics
    if num_ep == 0:
        obs_mean = obs_vec
        obs_std = np.std([obs_vec, obs_vec], axis=0)  # OSS: it's an array of zeros!
    else:
        obs_mean = np.mean([obs_mean, obs_vec], axis=0)
        obs_std = np.std(
            [obs_mean + obs_std, obs_mean - obs_std, obs_vec], axis=0
        )


# Re-scaling and other Statistics
obs_mean[:, 6:9] = obs_mean[:, 6:9] * l_star
obs_mean[:, 9:12] = obs_mean[:, 9:12] * l_star / t_star
obs_std[:, 6:9] = obs_std[:, 6:9] * l_star
obs_std[:, 9:12] = obs_std[:, 9:12] * l_star / t_star
prob_RVD = docked.sum() * 100 / num_episode_MCM

# Approach Corridor: truncated cone + cylinder
len_cut = np.sqrt((safety_radius**2) / np.square(np.tan(ang_corr)))
rad_kso = np.max(obs_mean[:, 6:9]) + len_cut
rad_entry = np.tan(ang_corr) * rad_kso
x_cone, z_cone = np.mgrid[-rad_entry:rad_entry:1000j, -rad_entry:rad_entry:1000j]
y_cone = np.sqrt((x_cone**2 + z_cone**2) / np.square(np.tan(ang_corr)))
y_cone = np.where(y_cone > rad_kso, np.nan, y_cone) - len_cut
y_cone = np.where(y_cone < 0, np.nan, y_cone)

# Plot full trajectory statistics
plt.close()
plt.figure()
ax = plt.axes(projection="3d")
mean_traj = ax.plot3D(
    obs_mean[:, 6], obs_mean[:, 7], obs_mean[:, 8], c="k", linewidth=2, label="Mean"
)
std_traj = ax.plot3D(np.nan, np.nan, np.nan, c="red", label="Std")
for _ in range(1000):
    std_traj = ax.plot3D(
        np.random.normal(obs_mean[:, 6], obs_std[:, 6]),
        np.random.normal(obs_mean[:, 7], obs_std[:, 7]),
        np.random.normal(obs_mean[:, 8], obs_std[:, 8]),
        c="r",
        alpha=0.005,
    )
start = ax.scatter(
    obs_mean[0, 6],
    obs_mean[0, 7],
    obs_mean[0, 8],
    color="orange",
    marker="s",
    label="Start",
)
stop = ax.scatter(
    obs_mean[-1, 6],
    obs_mean[-1, 7],
    obs_mean[-1, 8],
    color="cyan",
    marker="o",
    label="Stop",
)
goal = ax.scatter(0, 0, 0, color="green", marker="^", label="Goal")
ax.set_xlabel("$\delta x$ [m]", labelpad=15)
plt.xticks([0])
ax.plot_surface(x_cone, y_cone, z_cone, color="k", alpha=0.1)
plt.legend(loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.88))
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
ax.set_title(
    "RVD probability: %.1f %% \n Mean Final State: [%.3f m, %.3f m/s]"
    % (
        prob_RVD,
        np.linalg.norm(obs_mean[-1, 6:9]),
        np.linalg.norm(obs_mean[-1, 9:12]),
    ),
    y=1,
    pad=-3
)
plt.savefig("plots\MCM_Trajectory.pdf")  # Save
plt.show()
