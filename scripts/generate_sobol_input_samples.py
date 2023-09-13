from SALib.sample import saltelli
import numpy as np
import os
import shutil

# Begin User Input
prior_mean = np.zeros(3)
prior_cov = np.eye(3)
N = 2**15
input_procs = 1
output_procs = 200
repartition = True  # True if you want to repartition the samples
# End User Input
bounds = [[prior_mean[ii], prior_cov[ii, ii]] for ii in range(len(prior_mean))]

param_dict = {
    "num_vars": len(prior_mean),
    "names": ["x1", "x2", "x3"],
    "bounds": bounds,
    "dists": ["norm"] * len(prior_mean),
}

if repartition:
    input_parameter_samples = []
    for ii in range(input_procs):
        assert os.path.exists(
            "sobol_samples/sobol_input_samples_rank_{}.npy".format(ii)
        ), "sobol_input_samples.npy does not exist"
        input_parameter_samples.append(
            np.load("sobol_samples/sobol_input_samples_rank_{}.npy".format(ii))
        )
    assert (
        len(input_parameter_samples) == input_procs
    ), "Number of input parameter samples is incorrect"
    parameter_samples = np.vstack(input_parameter_samples)

    if os.path.exists("repart_sobol_samples"):
        shutil.rmtree("repart_sobol_samples")
        print("Removed existing repart_sobol_samples directory")
    os.makedirs("repart_sobol_samples")

    num_samples_per_proc = np.array(
        [int(parameter_samples.shape[0] / output_procs)] * output_procs
    )
    num_samples_per_proc[: parameter_samples.shape[0] % output_procs] += 1
    assert parameter_samples.shape[0] == np.sum(
        num_samples_per_proc
    ), "Number of samples per proc is incorrect"
    upper_idx = np.cumsum(num_samples_per_proc)
    lower_idx = upper_idx - num_samples_per_proc

    for ii in range(output_procs):
        np.save(
            "repart_sobol_samples/sobol_input_samples_rank_{}.npy".format(ii),
            parameter_samples[lower_idx[ii] : upper_idx[ii], :],
        )

else:
    assert input_procs == 1, "input_procs must be 1 if repartition is False"
    parameter_samples = saltelli.sample(param_dict, N)
    num_samples_per_proc = np.array(
        [int(parameter_samples.shape[0] / output_procs)] * output_procs
    )
    num_samples_per_proc[: parameter_samples.shape[0] % output_procs] += 1
    assert parameter_samples.shape[0] == np.sum(
        num_samples_per_proc
    ), "Number of samples per proc is incorrect"
    upper_idx = np.cumsum(num_samples_per_proc)
    lower_idx = upper_idx - num_samples_per_proc
    if os.path.exists("sobol_samples"):
        shutil.rmtree("sobol_samples")
        print("Removed existing sobol_samples directory")
    os.makedirs("sobol_samples")

    for ii in range(output_procs):
        np.save(
            "sobol_samples/sobol_input_samples_rank_{}.npy".format(ii),
            parameter_samples[lower_idx[ii] : upper_idx[ii], :],
        )
