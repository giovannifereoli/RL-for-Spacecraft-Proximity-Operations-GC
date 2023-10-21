# Import libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# REWARD
# Load data MLP
reward_data = pd.DataFrame.to_dict(pd.read_csv(
    r"C:\Users\giova\PycharmProjects\MetaRLopenAI\MLP\tensorboard\Rew2B.csv"
))
step = np.array(list(reward_data["Step"].values()))
reward = np.array(list(reward_data["Value"].values()))
# Load data LSTM
reward_dataLSTM = pd.DataFrame.to_dict(pd.read_csv(
    r"C:\Users\giova\PycharmProjects\MetaRLopenAI\MLP\tensorboard\Rew2LSTM.csv"
))
stepLSTM = np.array(list(reward_dataLSTM["Step"].values()))
rewardLSTM = np.array(list(reward_dataLSTM["Value"].values()))

# Plot Reward
plt.close()  # Initialize
plt.figure()
plt.semilogy(step, reward + 1.1, c="b", linewidth=2)
plt.semilogy(stepLSTM, rewardLSTM + 1.1, c="r", linewidth=2)
plt.xlim(0, 7*1e6)
plt.legend(["MLP", "LSTM"], loc="upper left")
plt.grid(True, which='both')
plt.xlabel("Learning Step [-]")
plt.ylabel("Mean Reward [-]")
plt.savefig("plots\MeanReward2.pdf")  # Save


