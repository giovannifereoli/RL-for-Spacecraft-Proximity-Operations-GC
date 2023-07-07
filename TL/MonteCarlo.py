# Import libraries
import numpy as np
from sb3_contrib import RecurrentPPO
from Environment import ArpodCrtbp
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt

# DEFINITIONS
# Data and initialization
m_star = 6.0458 * 1e24  # Kilograms
l_star = 3.844 * 1e8  # Meters
t_star = 375200  # Seconds

dt = 1
ToF = 600

rho_max = 2000
rhodot_max = 25

ang_corr = np.deg2rad(15)
safety_radius = 1
safety_vel = 0.1

max_thrust = 29620
mass = 21000
state_space = 14
actions_space = 3

x0t_state = np.array(
    [
        1.02206694e00,
        -2.61552389e-06,
        -1.82100000e-01,
        -3.34605533e-06,
        -1.03353155e-01,
        1.27335994e-05,
    ]
)  # 9:2 NRO - 50m after apolune, already corrected, rt = 399069639.7170633, vt = 105.88740083894766
x0r_state = np.array(
    [
        4.23387991e-11,
        2.61552389e-06,
        -1.61122476e-10,
        3.34605533e-06,
        -1.43029505e-10,
        -1.27335994e-05,
    ]
)
x0r_mass = np.array([mass / m_star])
x0_time_rem = np.array([ToF / t_star])
x0_vec = np.concatenate((x0t_state, x0r_state, x0r_mass, x0_time_rem))
x0_std_vec = np.absolute(
    np.concatenate(
        (
            np.zeros(6),
            2.5 * np.ones(3) / l_star,
            0.5 * np.ones(3) / (l_star / t_star),
            0.005 * x0r_mass,
            np.zeros(1),
        )
    )
)

# Define environment and model
env = ArpodCrtbp(
    max_time=ToF,
    dt=dt,
    rho_max=rho_max,
    rhodot_max=rhodot_max,
    x0=x0_vec,
    x0_std=x0_std_vec,
    ang_corr=ang_corr,
    safety_radius=safety_radius,
    safety_vel=safety_vel,
)
check_env(env)
model = RecurrentPPO(
    "MlpLstmPolicy",
    env,
    verbose=1,
    batch_size=2 * 32,
    n_steps=2 * 1920,
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
model = RecurrentPPO.load("ppo_recurrent")

# Trajectory propagation
num_episode_MCM = 200
num_ep = 0
collided = np.zeros(num_episode_MCM)
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
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=np.array([done]), deterministic=True
        )  # OSS: Episode start signals are used to reset the lstm states
        obs, rewards, done, info = env.step(action)

        # Saving
        obs_vec = np.vstack((obs_vec, env.scaler_reverse_observation(obs)))

        # Check collision (OSS: it happens during motion)
        if info.get("Episode success") == "collided":
            collided[num_ep] = 1

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
prob_collision = collided.sum() * 100 / num_episode_MCM
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
    "RVD probability: %.1f %% - Collision probability: %.1f %% \n Mean Final State: [%.3f m, %.3f m/s]"
    % (
        prob_RVD,
        prob_collision,
        np.linalg.norm(obs_mean[-1, 6:9]),
        np.linalg.norm(obs_mean[-1, 9:12]),
    ),
    y=1,
    pad=-3
)
plt.savefig("plots\MCM_Trajectory.pdf")  # Save
plt.show()
