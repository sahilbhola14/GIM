import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

sns.set_style("whitegrid")
plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=20)
plt.rc("lines", linewidth=3)
plt.rc("axes", labelpad=30, titlepad=20)
plt.rc("legend", fontsize=18, framealpha=1.0)
path = "/home/sbhola/Documents/CASLAB/GIM/examples/ignition_model/learning_model/campaign_results/campaign_1"
file_name = "estimated_individual_mutual_information.npy"
mi = np.load(os.path.join(path, file_name))

num_parameters = 3
xticklabels = [r"$\Theta_{}$".format(ii + 1) for ii in range(num_parameters)]

fig, axs = plt.subplots()
axs.bar(xticklabels, mi, color="b", edgecolor="k", linewidth=3, width=0.2)
axs.set_ylabel(r"$I(\Theta_i;Y\mid \Theta_{\sim i}, d)$")
axs.yaxis.set_minor_locator(AutoMinorLocator())
axs.set_ylim([0, 3])
plt.tight_layout()
plt.savefig("information_content.png")
plt.close()
