import numpy as np
import yaml
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import sys

plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=25)
plt.rc("lines", linewidth=3)
plt.rc("axes", labelpad=18, titlepad=20)
plt.rc("legend", fontsize=18, framealpha=1.0)
plt.rc("xtick.major", size=6)
plt.rc("xtick.minor", size=4)
plt.rc("ytick.major", size=6)
plt.rc("ytick.minor", size=4)

no_noise_data_path = "./ignition_delay_data"

# Load the yaml data
with open("./config.yaml", "r") as config_file:
    config_data = yaml.safe_load(config_file)

campaign_id = config_data["campaign_id"]
assert(campaign_id == int(sys.argv[1])), "Make sure the campaign id match"
model_noise_cov = config_data["model_noise_cov"]
data_file = config_data["data_file"]

# Load the data
no_noise_data = np.loadtxt(os.path.join(no_noise_data_path, data_file))
num_data_points = no_noise_data.shape[0]

training_data = np.copy(no_noise_data)

# Adding noise
noise = np.sqrt(model_noise_cov)*np.random.randn(num_data_points)
training_data[:, -1] = np.log(training_data[:, -1]) + noise
data_save_path = os.path.join("./campaign_results", "campaign_{0:d}/training_data.npy".format(campaign_id))
np.save(data_save_path, training_data)

fig_save_path = os.path.join("./campaign_results", "campaign_{0:d}/Figures/training_data.png".format(campaign_id))
plt.figure(figsize=(8, 5))
plt.scatter(1000/training_data[:, 0], np.log(no_noise_data[:, -1]), label="True", c="k", marker="s", s=50)
plt.scatter(1000/training_data[:, 0], training_data[:, -1], label="Data", c="r", marker="D", s=50)
plt.xlabel(r"1000/T [1/K]")
plt.ylabel(r"$\log(t_{ign})$")
plt.legend(loc="lower right")
plt.title("Training data")
plt.grid(color="k", alpha=0.5)
plt.tight_layout()
plt.savefig(fig_save_path)
plt.close()



# # Add noise
# spatial_resolution = no_noise_data.shape[1]
# noise = np.sqrt(model_noise_cov)*np.random.randn(num_data_points*spatial_resolution).reshape(num_data_points, -1)
# training_data = np.copy(no_noise_data)
# training_data[:, 1:] += noise[:, 1:]
# save_path = os.path.join("./campaign_results", "campaign_{0:d}/ytrain.npy".format(campaign_id))
# np.save(save_path, training_data)

# fig_save_path = os.path.join("./campaign_results", "campaign_{0:d}/Figures/training_data.png".format(campaign_id))
# plt.figure(figsize=(10, 5))
# fig, axs = plt.subplots(figsize=(10, 5))
# colors=["C{}".format(ii) for ii in range(num_data_points)]
# for idata in range(num_data_points):
#     axs.plot(time[idata, :]*1000, no_noise_data[idata, :], label=r"GRI-Mech3.0".format(equivalence_ratio[idata], initial_temperature[idata], initial_pressure[idata]), c="r")
#     axs.scatter(time[idata, :]*1000, training_data[idata, :], label=r"Data".format(equivalence_ratio[idata], initial_temperature[idata], initial_pressure[idata]), s=10, c="k")
# axs.legend(loc="lower right", framealpha=1.0)
# axs.set_xlabel("time [ms]")
# axs.set_ylabel("Temperature [K]")
# axs.grid()
# axs.xaxis.set_major_locator(MultipleLocator(1))
# axs.xaxis.set_minor_locator(MultipleLocator(0.5))
# plt.tight_layout()
# plt.savefig(fig_save_path)
# plt.close()
