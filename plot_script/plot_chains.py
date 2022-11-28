import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import os

plt.rc("font", family="serif", size=20)
plt.rc("text", usetex=True)
plt.rc("xtick", labelsize=20)
plt.rc("ytick", labelsize=20)
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

if num_parameters == 8:
    fig, axs = plt.subplots(4, 2, figsize=(20, 20))
    for i in range(num_parameters):
        ax = [i % 4, i // 4]
        axs[ax[0], ax[1]].plot(data[:, i], color="k")
        axs[ax[0], ax[1]].set_xlabel("Iteration")
        axs[ax[0], ax[1]].set_ylabel(theta[i])
elif num_parameters == 4:
    fig, axs = plt.subplots(4, 1, figsize=(20, 20))
    for i in range(num_parameters):
        ax = [i % 4, i // 4]
        axs[ax[0]].plot(data[:, i], color="k")
        axs[ax[0]].set_xlabel("Iteration")
        axs[ax[0]].set_ylabel(theta[i])
plt.tight_layout()
plt.savefig(os.path.join(path, "Figures/chains.png"), dpi=300)
