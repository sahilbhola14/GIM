import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import yaml

sys.path.append("../forward_model/")
from linear_gaussian import linear_gaussian


plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=25)
plt.rc("lines", linewidth=3)
plt.rc("axes", labelpad=18, titlepad=20)

with open("./config.yaml", "r") as config_file:
    config_data = yaml.safe_load(config_file)

campaign_id = config_data["campaign_id"]

assert campaign_id == int(sys.argv[1]), "Make sure the campaign id match"

total_samples = config_data["total_samples"]
true_model = linear_gaussian(spatial_res=total_samples)
true_prediction = true_model.compute_prediction(theta=true_model.true_theta)

# Subsampling
num_sub_samples = config_data["num_samples"]
assert num_sub_samples <= total_samples
sample_idx = np.arange(total_samples)
np.random.shuffle(sample_idx)
sample_idx = sample_idx[:num_sub_samples]


# Adding noise
model_noise_cov = config_data["model_noise_cov"]
noise = np.sqrt(model_noise_cov) * (np.random.randn(num_sub_samples))
training_data = np.zeros((num_sub_samples, 2))
training_data[:, 0] = true_model.xtrain[sample_idx]
training_data[:, 1] = true_prediction[sample_idx, 0] + noise


# Saving data
data_save_path = os.path.join(
    "./campaign_results", "campaign_{0:d}/training_data.npy".format(campaign_id)
)
sample_idx_save_path = os.path.join(
    "./campaign_results", "campaign_{0:d}/sample_idx.npy".format(campaign_id)
)
np.save(data_save_path, training_data)
np.save(sample_idx_save_path, sample_idx)


fig_save_path = os.path.join(
    "./campaign_results", "campaign_{0:d}/Figures/training_data.png".format(campaign_id)
)
plt.figure(figsize=(10, 5))
plt.plot(true_model.xtrain, true_prediction, label="True", c="r")
plt.scatter(training_data[:, 0], training_data[:, 1], label="Data", c="k")
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig(fig_save_path)
plt.close()
