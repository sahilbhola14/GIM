from SALib.analyze import sobol
import os
import numpy as np

# Begin User Input
root = "/home/sbhola/Documents/CASLAB/GIM/examples/ignition_model/learning_model"
sobol_input_sample_path = os.path.join(root, "sobol_samples")
sobol_output_sample_path = os.path.join(root, "campaign_results/campaign_6/SALib_Sobol")
eval_ranks = 5  # 1 Indexed
prior_mean = np.zeros(3)
prior_cov = np.eye(3)

# End User Input

bounds = [[prior_mean[ii], prior_cov[ii, ii]] for ii in range(len(prior_mean))]
param_dict = {
    "num_vars": len(prior_mean),
    "names": ["x1", "x2", "x3"],
    "bounds": bounds,
    "dists": ["norm"] * len(prior_mean),
}

stacked_predictions = []
for ii in range(eval_ranks):
    input_file = sobol_input_sample_path + "/sobol_input_samples_rank_{}.npy".format(ii)
    output_file = sobol_output_sample_path + "/sobol_output_samples_rank_{}.npy".format(
        ii
    )

    assert os.path.exists(output_file), "Output file {} does not exist".format(
        output_file
    )

    model_predictions = np.load(output_file)
    print(model_predictions.shape)
    num_data_pts, spatial_res, _ = model_predictions.shape
    stacked_predictions.append(model_predictions[:, 0, :].T)

stacked_predictions = np.vstack(stacked_predictions)
# plt.figure()
# plt.plot(1/np.arange(1, stacked_predictions.shape[-1] + 1), stacked_predictions.T)
# plt.show()
# breakpoint()

# Compute Sobol Indices

first_order_samples = np.zeros((param_dict["num_vars"], num_data_pts))
second_order_samples = np.zeros((param_dict["num_vars"], num_data_pts))

for idim in range(num_data_pts):
    Si = sobol.analyze(
        param_dict,
        stacked_predictions[:, idim],
        calc_second_order=True,
        print_to_console=False,
    )
    first_order_samples[:, idim] = Si["S1"]
    second_order_samples[:, idim] = np.array(
        [Si["S2"][0, 1], Si["S2"][0, 2], Si["S2"][1, 2]]
    )

first_order = np.mean(first_order_samples, axis=1)
second_order = np.mean(second_order_samples, axis=1)
print("First Order Sobol Indices: {}".format(first_order))
print("Second Order Sobol Indices: {}".format(second_order))
