import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
import os

sns.set_style("whitegrid")
plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=20)
plt.rc("lines", linewidth=3)
plt.rc("axes", labelpad=30, titlepad=20)
plt.rc("legend", fontsize=18, framealpha=1.0)
path = (
    "/home/sbhola/Documents/CASLAB/GIM/examples/ignition_model/"
    "learning_model/campaign_results/campaign_1"
)
file_name = "estimated_pair_mutual_information.npy"
cmi = np.load(os.path.join(path, file_name))

num_parameters = 3
com = list(combinations(np.arange(num_parameters), 2))
matrix = np.nan * np.eye(num_parameters)

for kk, ipair in enumerate(com):
    ii, jj = ipair
    matrix[ii, jj] = cmi[kk]
    matrix[jj, ii] = cmi[kk]
print(matrix)
xticklabels = [r"$\Theta_{}$".format(ii + 1) for ii in range(num_parameters)]
colormap = "inferno"
num_matrix = matrix.shape[-1]

fig, axs = plt.subplots(1, 1, edgecolor="k", figsize=(10, 10))
g = sns.heatmap(
    matrix,
    cmap=colormap,
    annot=True,
    square=True,
    linewidth=3,
    linecolor="k",
    fmt=".3e",
    cbar_kws={"aspect": 15, "shrink": 0.8},
    xticklabels=xticklabels,
    yticklabels=xticklabels,
    annot_kws={"bbox": {"facecolor": "w"}, "color": "k"},
    ax=axs,
    vmin=0,
    vmax=1.2,
)
for _, spine in g.spines.items():
    spine.set_visible(True)
    spine.set_linewidth(3)
    spine.set_color("k")
axs.set_title(r"$I(\Theta_i;\Theta_j\mid Y, \Theta_{\sim i, j}, d)$")
plt.tight_layout()
plt.savefig("parameter_dependence.png")
plt.close()
