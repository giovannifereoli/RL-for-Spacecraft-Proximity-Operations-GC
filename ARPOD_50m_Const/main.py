# Import libraries
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy
from Environment import ArpodCrtbp
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt

# TRAINING
# Data and initialization
m_star = 6.0458 * 1e24  # Kilograms
l_star = 3.844 * 1e8  # Meters
t_star = 375200  # Seconds

dt = 0.5  # TODO: prova un sec, prova batch diversi, prova a velocizzare.... se non hai tempo l'importante è che converga
ToF = 30

rho_max = 100
rhodot_max = 6

max_thrust = 29620
mass = 21000
state_space = 14
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
x0_vec = np.concatenate((x0t_state, x0r_state, x0r_mass, x0_time_rem))
x0_std_vec = np.absolute(
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
    x0=x0_vec,
    x0_std=x0_std_vec,
)
check_env(env)
model = RecurrentPPO(
    "MlpLstmPolicy",
    env,
    verbose=1,
    batch_size=2*32,
    n_steps=2*1920,
    n_epochs=10,
    learning_rate=0.0001,  # OSS: ormai sono abbastanza sicuro con questi HP.
    gamma=0.99,
    gae_lambda=1,  # TODO: prova 0.00005 lr su questo
    clip_range=0.1,
    max_grad_norm=0.1,
    ent_coef=1e-4,
    # policy_kwargs=dict(enable_critic_lstm=False, optimizer_kwargs=dict(weight_decay=1e-5)),
    tensorboard_log="./tensorboard/"
)

print(model.policy)

# Start learning
model.learn(total_timesteps=3000000, progress_bar=True)  # TODO: rifletti piu sul sgnificato di metaRL e come metterlo alla prova

# Evaluation and saving
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20, warn=False)
print(mean_reward)
model.save("ppo_recurrent")

# TESTING
# Remove to demonstrate saving and loading
del model

# Loading model and reset environment
model = RecurrentPPO.load("ppo_recurrent")
obs = env.reset()

# Trajectory propagation
lstm_states = None
done = True
obs_vec = env.scaler_reverse_observation(obs)
rewards_vec = np.array([])
actions_vec = np.zeros(3)

while True:
    # Action sampling and propagation
    action, lstm_states = model.predict(
        obs, state=lstm_states, episode_start=np.array([done]), deterministic=True
    )  # OSS: Episode start signals are used to reset the lstm states
    obs, rewards, done, info = env.step(action)

    # Saving
    actions_vec = np.vstack((actions_vec, env.scaler_reverse_action(action)))
    obs_vec = np.vstack((obs_vec, env.scaler_reverse_observation(obs)))
    rewards_vec = np.append(rewards_vec, rewards)

    # Stop propagation
    if done:
        break

# PLOTS
# Plotted quantities
position = obs_vec[:, 6:9] * l_star
velocity = obs_vec[:, 9:12] * l_star / t_star
mass = obs_vec[:, 12] * m_star
thrust = actions_vec * (m_star * l_star / t_star**2)
t = np.linspace(0, ToF, int(ToF / dt) + 1)

# Plot full trajectory ONCE
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

# Plot mass usage
plt.close()  # Initialize
plt.figure()
plt.plot(t, mass, c="r", linewidth=2)
plt.grid(True)
plt.xlabel("Time [s]")
plt.ylabel("Mass [kg]")
plt.savefig("plots\Mass.pdf")  # Save

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
