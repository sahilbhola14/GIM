import numpy as np
import matplotlib.pyplot as plt

# import seaborn as sns
# import pandas as pd
import sys
import os

plt.rc("font", family="serif", size=25)
plt.rc("text", usetex=True)
plt.rc("xtick", labelsize=25)
plt.rc("ytick", labelsize=25)
plt.rc("axes", labelpad=25, titlepad=20)
plt.rc("lines", linewidth=3)

num_parameters = 4
num_procs = 8
theta = [r"$\Theta_{}$".format(i) for i in range(1, num_parameters + 1)]
campaign = "campaign_{}".format(sys.argv[1])
path = "./campaign_results/{}/".format(campaign)

data = []
for ichain in range(num_procs):
    data_path = os.path.join(path, "burned_samples_rank_{}_.npy".format(ichain))
    data.append(np.load(data_path))
data = np.concatenate(data, axis=0)

fig, axs = plt.subplots(num_parameters, 1, figsize=(20, 20))
for i in range(num_parameters):
    axs[i].plot(data[:, i], color="k")
    axs[i].set_xlabel("Iteration")
    axs[i].set_ylabel(theta[i])

plt.tight_layout()
plt.savefig(os.path.join(path, "Figures/chains.png"), dpi=300)
