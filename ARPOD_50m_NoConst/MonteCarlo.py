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

dt = 0.5
ToF = 30

rho_max = 100
rhodot_max = 6

max_thrust = 29620
mass = 21000
state_space = 13
actions_space = 3

x0t_state = np.array(
    [
        1.02206694e00,
        -1.33935003e-07,
        -1.82100000e-01,
        -1.71343849e-07,
        -1.03353155e-01,
        6.52058535e-07,
    ]
)  # 9:2 NRO - 50m after apolune, already corrected, rt = 399069639.7170633, vt = 105.88740083894766
x0r_state = np.array(
    [
        1.11022302e-13,
        1.33935003e-07,
        -4.22495372e-13,
        1.71343849e-07,
        -3.75061093e-13,
        -6.52058535e-07,
    ]
)
x0r_mass = np.array([mass / m_star])
x0_vec = np.concatenate((x0t_state, x0r_state, x0r_mass))
x0_std_vec = np.absolute(
    np.concatenate(
        (
            np.zeros(6),
            2.5 * np.ones(3) / l_star,
            0.5 * np.ones(3) / (l_star / t_star),
            0.005 * x0r_mass,
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
)
check_env(env)
model = RecurrentPPO(
    "MlpLstmPolicy",
    env,
    verbose=1,
    batch_size=2 * 32,
    n_steps=2 * 1920,
    n_epochs=10,
    learning_rate=0.0005,
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
num_episode_MCM = 1000
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
        )  # TODO: check questi conti


# Re-scaling and other Statistics
obs_mean[:, 6:9] = obs_mean[:, 6:9] * l_star
obs_mean[:, 9:12] = obs_mean[:, 9:12] * l_star / t_star
obs_std[:, 6:9] = obs_std[:, 6:9] * l_star
obs_std[:, 9:12] = obs_std[:, 9:12] * l_star / t_star
prob_collision = collided.sum() * 100 / num_episode_MCM
prob_RVD = docked.sum() * 100 / num_episode_MCM

# Approach Corridor
rad_kso = 50
ang_corr = np.deg2rad(10)
rad_entry = np.tan(ang_corr) * rad_kso
x_cone, y_cone = np.mgrid[-rad_entry:rad_entry:1000j, -rad_entry:rad_entry:1000j]
z_cone = np.sqrt((x_cone**2 + y_cone**2) / np.square(np.tan(ang_corr)))
z_cone = np.where(z_cone > rad_kso, np.nan, z_cone)

# Plot full trajectory statistics
plt.close()
plt.figure()
ax = plt.axes(projection="3d")
ax.plot3D(
    obs_mean[:, 6],
    obs_mean[:, 7],
    obs_mean[:, 8],
    c="k",
    linewidth=2,
)
start = ax.scatter(
    obs_mean[0, 6],
    obs_mean[0, 7],
    obs_mean[0, 8],
    color="blue",
    marker="s",
)
stop = ax.scatter(
    obs_mean[-1, 6],
    obs_mean[-1, 7],
    obs_mean[-1, 8],
    color="red",
    marker="o",
)
goal = ax.scatter(0, 0, 0, color="green", marker="^")
plt.legend(
    (start, stop, goal),
    ("Start", "Stop", "Goal"),
    scatterpoints=1,
    loc="upper right",
)
ax.set_xlabel("$\delta x$ [m]", labelpad=15)
plt.xticks([0])
for _ in range(1000):
    ax.plot3D(
        np.random.normal(obs_mean[:, 6], obs_std[:, 6]),
        np.random.normal(obs_mean[:, 7], obs_std[:, 7]),
        np.random.normal(obs_mean[:, 8], obs_std[:, 8]),
        c="r",
        alpha=0.01,
    )
ax.plot_surface(x_cone, z_cone, y_cone, color="k", alpha=0.1)
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
    "RVD probability %.1f %% - Collision probability %.1f %%"
    % (prob_RVD, prob_collision)
)
plt.show()
# plt.savefig("plots\Trajectory.pdf")  # Save
