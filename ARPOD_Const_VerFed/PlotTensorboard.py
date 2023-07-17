# Import libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# REWARD
# Load data
reward_data = pd.DataFrame.to_dict(pd.read_csv(
    r"/ARPOD_50m_Const/tensorboard/RecurrentPPO_2/run-.-tag-rollout_ep_rew_mean.csv"
))
step = np.array(list(reward_data["Step"].values()))
reward = np.array(list(reward_data["Value"].values()))

# Plot Reward
plt.close()  # Initialize
plt.figure()
plt.plot(step, reward, c="b", linewidth=2)
plt.grid(True)
plt.xlabel("Learning Step [-]")
plt.ylabel("Mean Reward [-]")
plt.savefig("plots\MeanReward.pdf")  # Save


