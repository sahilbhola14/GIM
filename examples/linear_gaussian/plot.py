import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=25)

num_parameters = 3

parameter_pair = np.array(list(combinations(np.arange(num_parameters), 2)))


def plot_pair_parameter_mutual_information():
    """Function plots the pair parameter mutual information"""

    pair_mi = np.load("estimated_pair_mutual_information.npy")
    pair_mi_tensor = np.nan*np.ones((num_parameters, num_parameters))
    pair_mi_tensor[parameter_pair[:, 0], parameter_pair[:, 1]] = pair_mi
    pair_mi_tensor[parameter_pair[:, 1], parameter_pair[:, 0]] = pair_mi

    true_pair_mi = np.array(
        [-4.263256414560601e-14, 0.892374332166213, -9.947598300641403e-14])
    true_pair_mi_tensor = np.nan*np.ones((num_parameters, num_parameters))
    true_pair_mi_tensor[parameter_pair[:, 0],
                        parameter_pair[:, 1]] = true_pair_mi
    true_pair_mi_tensor[parameter_pair[:, 1],
                        parameter_pair[:, 0]] = true_pair_mi

    label = [r'$\theta_{}$'.format(ii+1) for ii in range(num_parameters)]

    # fig = plt.figure(figsize=(20, 10))
    # plt.subplot(1, 2, 1)
    # res_true = sns.heatmap(true_pair_mi_tensor, xticklabels=label, yticklabels=label, linewidths=5, linecolor="k", square=True, cmap="jet", annot=True,
    #                        annot_kws={'fontsize': 30, 'fontstyle': 'normal', 'color': 'k', 'alpha': 1.0,
    #                                   'verticalalignment': 'center', 'backgroundcolor': 'w'}, vmin=0, vmax=3, cbar=False)

    # plt.subplot(1, 2, 2)
    # res = sns.heatmap(pair_mi_tensor, xticklabels=label, yticklabels=label, linewidths=5, linecolor="k", square=True, cmap="jet", annot=True,
    #                   annot_kws={'fontsize': 30, 'fontstyle': 'normal', 'color': 'k', 'alpha': 1.0,
    #                              'verticalalignment': 'center', 'backgroundcolor': 'w'}, vmin=0, vmax=3,
    #                   cbar_kws={'shrink': 0.9, 'pad': 0.08})

    # for _, spine in res.spines.items():
    #     spine.set_visible(True)
    #     spine.set_linewidth(5)

    # for _, spine in res_true.spines.items():
    #     spine.set_visible(True)
    #     spine.set_linewidth(5)

    # plt.suptitle(r"$I(\Theta_i;\Theta_j\mid Y;\Theta_{-i, j})$")

    # plt.tight_layout()
    # plt.savefig("parameter_dependence.png")
    # plt.show()
    color_map = "inferno"
    fig, axs = plt.subplots(figsize=(20, 10), ncols=3,
                            gridspec_kw=dict(width_ratios=[2, 2, 0.2]))

    sns.heatmap(true_pair_mi_tensor, xticklabels=label, yticklabels=label, linewidth=5, linecolor="k", annot=True, cbar=False, ax=axs[0], vmin=0, vmax=3, square=True,
                annot_kws={'fontsize': 30, 'fontstyle': 'normal', 'color': 'k', 'alpha': 1.0,
                           'verticalalignment': 'center', 'backgroundcolor': 'w'}, cmap=color_map)

    sns.heatmap(pair_mi_tensor, annot=True, xticklabels=label, yticklabels=label, linewidth=5, linecolor="k", cbar=False, ax=axs[1], vmin=0, vmax=3, square=True,
                annot_kws={'fontsize': 30, 'fontstyle': 'normal', 'color': 'k', 'alpha': 1.0,
                           'verticalalignment': 'center', 'backgroundcolor': 'w'}, cmap=color_map)

    for _, spine in axs[0].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(5)

    for _, spine in axs[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(5)

    fig.colorbar(axs[1].collections[0], cax=axs[2], aspect=50)

    axs[0].set_title(
        r"$I(\Theta_i;\Theta_j\mid Y, \Theta_{-i, j})_{true}$", pad=20)
    axs[1].set_title(
        r"$I(\Theta_i;\Theta_j\mid Y, \Theta_{-i, j})_{estimate}$", pad=20)

    plt.tight_layout()
    plt.savefig("parameter_dependence.png")
    plt.show()


def plot_individual_parameter_mutual_information():
    """Function plots the individual parameter mutual information"""
    # Load the data
    individual_parameter_mutual_information = np.load(
        "estimated_individual_mutual_information.npy")
    true_parameter_mutual_information = np.array(
        [2.91603997, 2.67148921, 2.51401019])

    # Set width of bar
    barWidth = 0.25
    fig, axs = plt.subplots(figsize=(12, 6))

    br1 = np.arange(num_parameters)
    br2 = [x + barWidth for x in br1]

    # Make the plot
    axs.bar(br1, true_parameter_mutual_information, color='r', width=barWidth,
            edgecolor='k', linewidth=4, label=r'True')
    axs.bar(br2, individual_parameter_mutual_information, color='b', width=barWidth,
            edgecolor='k', linewidth=4, label=r'Estimate')

    # Adding Xticks
    axs.set_xticks([r+0.5*barWidth for r in range(num_parameters)])
    axs.set_xticklabels(['$\Theta_{}$'.format(ii+1)
                        for ii in range(num_parameters)])
    axs.set_xlabel(r'$\Theta_{i}$', labelpad=20)
    axs.set_ylabel('$I(\Theta_{i};Y\mid\Theta_{-i})$',
                   fontweight='bold', labelpad=20)
    axs.yaxis.set_minor_locator(AutoMinorLocator())
    axs.grid(alpha=0.5, lw=1.2, which="major", color="k")
    axs.grid(alpha=0.2, lw=0.8, which="minor", color="k")
    axs.legend()
    plt.tight_layout()
    plt.savefig("information_content.png")
    plt.show()


def main():
    # plot_individual_parameter_mutual_information()
    plot_pair_parameter_mutual_information()


if __name__ == "__main__":
    main()
