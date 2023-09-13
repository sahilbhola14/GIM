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

df = pd.DataFrame(data, columns=theta)

sns.pairplot(
    df,
    diag_kind="kde",
    corner=True,
    # kind="kde",
    plot_kws={"color": "k", "alpha": 0.3},
    diag_kws={"color": "k"},
)
plt.tight_layout()
plt.savefig(os.path.join(path, "Figures/histogram.png"), dpi=300)
plt.close()
