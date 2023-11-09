# Import libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# REWARD
# Load data MLP
reward_data = pd.DataFrame.to_dict(pd.read_csv(
    r"C:\Users\giova\PycharmProjects\MetaRLopenAI\MLPconstAng\tensorboard\Rew.csv"
))
step = np.array(list(reward_data["Step"].values()))
reward = np.array(list(reward_data["Value"].values()))
# Load data LSTM
reward_dataLSTM = pd.DataFrame.to_dict(pd.read_csv(
    r"C:\Users\giova\PycharmProjects\MetaRLopenAI\MLPconstAng\tensorboard\RewLSTM3.csv"
))
stepLSTM = np.array(list(reward_dataLSTM["Step"].values()))
rewardLSTM = np.array(list(reward_dataLSTM["Value"].values()))


# Plot Reward
plt.close()  # Initialize
plt.figure()
plt.semilogy(step, reward + 0.65, c="b", linewidth=2)
plt.semilogy(stepLSTM, rewardLSTM + 0.65, c="r", linewidth=2)
plt.legend(["MLP", "LSTM"], loc="upper left")
plt.grid(True, which='both')
plt.xlabel("Learning Step [-]")
plt.ylabel("Mean Reward [-]")
plt.savefig("plots\MeanReward.pdf")  # Save


