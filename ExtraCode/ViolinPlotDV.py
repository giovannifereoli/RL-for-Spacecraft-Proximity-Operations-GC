# Plot for DV explanation
# Libraries
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # You need seaborn for violin plots


# Data
opt_50 = 5.763
opt_200 = 11.153
ocp_50 = np.random.normal(5.763, 0.684, size=10000) / opt_50
ocp_200 = np.random.normal(11.153, 0.772, size=10000) / opt_200
dv_50_lstm = (np.random.normal(6.350, 0.110, size=10000) - opt_50) / opt_50
dv_200_lstm = (np.random.normal(11.981, 0.495, size=10000) - opt_200) / opt_200
dv_50_mlp = (np.random.normal(9.930, 0.262, size=10000) - opt_50) / opt_50
dv_200_mlp = (np.random.normal(18.669, 0.537, size=10000) - opt_200) / opt_200
pos = [1, 2, 3, 4]
data = [dv_50_lstm, dv_200_lstm, dv_50_mlp, dv_200_mlp]

# Create a figure instance
fig = plt.figure()
ax = plt.subplot(111)

# Create the boxplot
bp = plt.violinplot(data, pos, showmeans=True)
bp = plt.grid(True)
ax.set_xticks(pos)
ax.set_ylabel("$(\Delta V - \Delta V_{OCP}) / \Delta V_{OCP}$")
ax.set_xticklabels(["LSTM\n50m", "LSTM\n200m", "MLP\n50m", "MLP\n200m"])
plt.show()
