# Import libraries
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy
from EnvironmentThesis import CustomEnv
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt

# TRAINING
# Define environment and model
env = CustomEnv()
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

# Start learning
model.learn(4000000, progress_bar=True)

# Evaluation and saving
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20, warn=False)
print(mean_reward)
model.save("ppo_recurrent")

# TESTING
# Data
m_star = 6.0458 * 1e24  # Kilograms
l_star = 3.844 * 1e8  # Meters
t_star = 375200  # Seconds
dt = 0.5
ToF = 30
max_thrust = 29620
state_space = 13
actions_space = 3

# Remove to demonstrate saving and loading
del model

# Loading model and reset environment
model = RecurrentPPO.load("ppo_recurrent")
obs = env.reset()

# Cell and hidden state of the LSTM loading
lstm_states = None
num_envs = 1  # OSS: possibility to implement vectorized environments

# Trajectory propagation
episode_starts = np.ones(
    (num_envs,), dtype=bool
)  # OSS: Episode start signals are used to reset the lstm states
episode_ends = np.zeros((num_envs,), dtype=bool)
obs_vec = np.array([])
rewards_vec = np.array([])
actions_vec = np.array([])

while episode_ends == np.zeros((num_envs,), dtype=bool):
    # Action sampling and propagation
    action, lstm_states = model.predict(
        obs, state=lstm_states, episode_start=episode_starts, deterministic=True
    )
    obs, rewards, dones, info = env.step(action)
    episode_starts = dones
    if episode_starts == np.ones((num_envs,), dtype=bool):
        episode_ends = np.ones((num_envs,), dtype=bool)

    # Saving
    actions_vec = np.append(actions_vec, max_thrust * action / np.linalg.norm(np.array([1, 1, 1])))
    obs_vec = np.append(obs_vec, env.scaler_reverse(obs))
    rewards_vec = np.append(rewards_vec, rewards)

# PLOTS
# Re-organize arrays
obs_vec = np.transpose(np.reshape(obs_vec, (state_space, int(len(obs_vec) / state_space)), order='F'))
actions_vec = np.transpose(np.reshape(actions_vec, (actions_space, int(len(actions_vec) / actions_space)), order='F'))

# Plotted quantities
position = obs_vec[:, 6:9] * l_star
velocity = obs_vec[:, 9:12] * l_star / t_star
thrust = actions_vec
t = np.linspace(0, ToF, int(ToF / dt))
t = t[0:len(position)]

# Plot full trajectory
plt.close()
plt.figure()
ax = plt.axes(projection="3d")
ax.plot3D(
    position[:, 0],
    position[:, 1],
    position[:, 2],
    c="k",
    linewidth=2,
)
start = ax.scatter(
    position[0, 0],
    position[0, 1],
    position[0, 2],
    color="blue",
    marker="s",
)
stop = ax.scatter(
    position[-1, 0],
    position[-1, 1],
    position[-1, 2],
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
ax.set_ylabel("$\delta y$ [m]", labelpad=10)
ax.zaxis.set_rotate_label(False)
ax.set_zlabel("$\delta z$ [m]", labelpad=10, rotation=90)
# plt.locator_params(axis="x", nbins=1)
plt.locator_params(axis="y", nbins=6)
plt.locator_params(axis="z", nbins=6)
ax.xaxis.pane.set_edgecolor("black")
ax.yaxis.pane.set_edgecolor("black")
ax.zaxis.pane.set_edgecolor("black")
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
# ax.set_aspect("auto")
ax.view_init(elev=0, azim=0)
plt.savefig("plots\Trajectory.pdf")  # Save

# Plot relative velocity norm
plt.close()  # Initialize
plt.figure()
plt.plot(t, np.linalg.norm(velocity, axis=1), c="b", linewidth=2)
plt.grid(True)
plt.xlabel("Time [s]")
plt.ylabel("Velocity [m/s]")
plt.savefig("plots\Velocity.pdf")  # Save

# Plot relative position
plt.close()  # Initialize
plt.figure()
plt.plot(t, np.linalg.norm(position, axis=1), c="g", linewidth=2)
plt.grid(True)
plt.xlabel("Time [s]")
plt.ylabel("Position [m]")
plt.savefig("plots\Position.pdf")  # Save

# Plot CoM control action
plt.close()
plt.figure()
plt.plot(
    t,
    thrust[:, 0],
    c="g",
    linewidth=2,
)
plt.plot(
    t,
    thrust[:, 1],
    c="b",
    linewidth=2,
)
plt.plot(
    t,
    thrust[:, 2],
    c="r",
    linewidth=2,
)
plt.legend(["$T_x$", "$T_y$", "$T_z$"], loc="upper right")
plt.grid(True)
plt.xlabel("Time [s]")
plt.ylabel("Thrust [N]")
plt.xlim(t[0], t[-1])
plt.savefig("plots\Thrust.pdf", bbox_inches="tight")  # Save

# TODO: maybe last plots can be written as MCM (con mean e standard deviation sui plot)
# TODO: sistema GitHub
