import numpy as np

# Begin user input
num_parameters = 3
num_ranks = 4
num_global_outer_samples = 1200
data_shape = (1, 100)
# End user input


batch_size = num_global_outer_samples // num_ranks

# Read the sobol denominator
sobol_denominator = np.load("sobol_denominator.npy")

# Compute the first order sobol indices
sobol_numerator = np.zeros((data_shape + (num_parameters,)))
for iparam in range(num_parameters):
    inner_expectation = np.zeros((data_shape + (num_global_outer_samples,)))
    for irank in range(num_ranks):
        data = np.load(
            "first_order_inner_expectation_param_{}_rank_{}.npy".format(
                iparam + 1, irank
            )
        )
        inner_expectation[:, :, irank * batch_size : (irank + 1) * batch_size] = data

    sobol_numerator[:, :, iparam] = np.var(inner_expectation, axis=2)

ratio = sobol_numerator / sobol_denominator[:, :, np.newaxis]
sobol_index = np.mean(ratio, axis=(0, 1))
print("First order sobol index: {}".format(sobol_index))

# Compute the total effect sobol indices
sobol_numerator = np.zeros((data_shape + (num_parameters,)))
for iparam in range(num_parameters):
    inner_expectation = np.zeros((data_shape + (num_global_outer_samples,)))
    for irank in range(num_ranks):
        data = np.load(
            "total_effect_inner_expectation_param_{}_rank_{}.npy".format(
                iparam + 1, irank
            )
        )
        inner_expectation[:, :, irank * batch_size : (irank + 1) * batch_size] = data

    sobol_numerator[:, :, iparam] = np.var(inner_expectation, axis=2)

ratio = sobol_numerator / sobol_denominator[:, :, np.newaxis]
sobol_index = np.mean(ratio, axis=(0, 1))
print("Total effect sobol index: {}".format(1 - sobol_index))
