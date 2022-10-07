import numpy as np
import yaml
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=25)
plt.rc("lines", linewidth=3)
plt.rc("axes", labelpad=18, titlepad=20)

no_noise_data_path = "./gri_data"

# Load the yaml data
with open("./config.yaml", "r") as config_file:
    config_data = yaml.safe_load(config_file)

campaign_id = config_data["campaign_id"]
equivalence_ratio = config_data["equivalence_ratio"]
initial_temperature = config_data["initial_temperature"]
initial_pressure = config_data["initial_pressure"]
model_noise_cov = config_data["model_noise_cov"]

assert(len(equivalence_ratio) == len(initial_temperature) == len(initial_pressure)), "Provide input config for all cases"
num_data_points = len(equivalence_ratio)
no_noise_data = []
time = []
for ii in range(num_data_points):
    no_noise_data_file = os.path.join(no_noise_data_path, "gri_data_phi_{}_t_{}_p_{}.npy".format(equivalence_ratio[ii], initial_temperature[ii], initial_pressure[ii]))
    data = np.load(no_noise_data_file)
    time.append(data[:, 0])
    no_noise_data.append(data[:, 1])

no_noise_data = np.array(no_noise_data)
time = np.array(time)
# Load the no noise data

# Add noise
spatial_resolution = no_noise_data.shape[1]
noise = np.sqrt(model_noise_cov)*np.random.randn(num_data_points*spatial_resolution).reshape(num_data_points, -1)
training_data = np.copy(no_noise_data)
training_data[:, 1:] += noise[:, 1:]
save_path = os.path.join("./campaign_results", "campaign_{0:d}/ytrain.npy".format(campaign_id))
np.save(save_path, training_data)

fig_save_path = os.path.join("./campaign_results", "campaign_{0:d}/Figures/training_data.png".format(campaign_id))
plt.figure(figsize=(10, 5))
fig, axs = plt.subplots(figsize=(10, 5))
colors=["C{}".format(ii) for ii in range(num_data_points)]
for idata in range(num_data_points):
    axs.plot(time[idata, :]*1000, no_noise_data[idata, :], label=r"GRI-Mech3.0".format(equivalence_ratio[idata], initial_temperature[idata], initial_pressure[idata]), c="r")
    axs.scatter(time[idata, :]*1000, training_data[idata, :], label=r"Data".format(equivalence_ratio[idata], initial_temperature[idata], initial_pressure[idata]), s=10, c="k")
axs.legend(loc="lower right", framealpha=1.0)
axs.set_xlabel("time [ms]")
axs.set_ylabel("Temperature [K]")
axs.grid()
axs.xaxis.set_major_locator(MultipleLocator(1))
axs.xaxis.set_minor_locator(MultipleLocator(0.5))
plt.tight_layout()
plt.savefig(fig_save_path)
plt.close()
