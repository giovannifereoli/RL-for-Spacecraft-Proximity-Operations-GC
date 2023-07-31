# Import libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# REWARD
# Load data
reward_data = pd.DataFrame.to_dict(pd.read_csv(
    r"C:\Users\giova\PycharmProjects\MetaRLopenAI\MLP\tensorboard\Rew.csv"
))
step = np.array(list(reward_data["Step"].values()))
reward = np.array(list(reward_data["Value"].values()))

# Plot Reward
plt.close()  # Initialize
plt.figure()
plt.semilogy(step, reward + 1.15, c="b", linewidth=2)
plt.grid(True)
plt.xlabel("Learning Step [-]")
plt.ylabel("Mean Reward [-]")
plt.savefig("plots\MeanReward.pdf")  # Save


