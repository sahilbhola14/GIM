import numpy as np
import os

# Begin user input
num_parameters = None
specific_parameters = [1]  # 1 indexed
num_ranks = 180
num_global_outer_samples = 9000
data_shape = (1, 4)
compute_first_effect_sobol = True
compute_total_effect_sobol = False
# End user input

if num_parameters is None:
    assert specific_parameters is not None, "Provide a specific parameter to evaluate"

if specific_parameters is None:
    assert num_parameters is not None, "Provide total number of parameters"


batch_size = num_global_outer_samples // num_ranks

# Read the sobol denominator
sobol_denominator = np.load("sobol_denominator.npy")

# Compute the first order sobol indices
if num_parameters is None:
    sobol_numerator = np.zeros((data_shape + (len(specific_parameters),)))
    total_parameters = len(specific_parameters)
    parameter_list = np.array(specific_parameters) - 1
else:
    sobol_numerator = np.zeros((data_shape + (num_parameters,)))
    total_parameters = num_parameters
    parameter_list = np.arange(num_parameters)

unavailable_ranks = []
if compute_first_effect_sobol:
    for ii in range(total_parameters):

        iparam = parameter_list[ii]
        unavailable_ranks = []

        # Check if file is available
        for irank in range(num_ranks):
            file_name = "first_order_inner_expectation_param_{}_rank_{}.npy".format(
                iparam + 1, irank
            )

            data_ = os.path.exists(file_name)
            if os.path.exists(file_name) is False:
                unavailable_ranks.append(irank)

        total_available_outer_samples = num_global_outer_samples - (
            len(unavailable_ranks) * batch_size
        )
        inner_expectation = np.zeros((data_shape + (total_available_outer_samples,)))
        print("unavailable ranks : {}".format(unavailable_ranks))
        available_ranks = np.delete(np.arange(num_ranks), unavailable_ranks)

        for jj, irank in enumerate(available_ranks):
            file_name = "first_order_inner_expectation_param_{}_rank_{}.npy".format(
                iparam + 1, irank
            )
            data = np.load(file_name)

            inner_expectation[:, :, jj * batch_size : (jj + 1) * batch_size] = data

        sobol_numerator[:, :, ii] = np.var(inner_expectation, axis=2)

    ratio = sobol_numerator / sobol_denominator[:, :, np.newaxis]
    sobol_index = np.mean(ratio, axis=(0, 1))
    print("First order sobol index: {}".format(sobol_index))

# Compute the total effect sobol indices
if compute_total_effect_sobol:
    unavailable_ranks = []

    for ii in range(total_parameters):

        iparam = parameter_list[ii]
        unavailable_ranks = []

        # Check if file is available
        for irank in range(num_ranks):
            file_name = "total_effect_inner_expectation_param_{}_rank_{}.npy".format(
                iparam + 1, irank
            )

            data_ = os.path.exists(file_name)
            if os.path.exists(file_name) is False:
                unavailable_ranks.append(irank)

        total_available_outer_samples = num_global_outer_samples - (
            len(unavailable_ranks) * batch_size
        )
        inner_expectation = np.zeros((data_shape + (total_available_outer_samples,)))

        available_ranks = np.delete(np.arange(num_ranks), unavailable_ranks)

        for ii, irank in enumerate(available_ranks):
            file_name = "first_order_inner_expectation_param_{}_rank_{}.npy".format(
                iparam + 1, irank
            )
            data = np.load(file_name)

            inner_expectation[:, :, ii * batch_size : (ii + 1) * batch_size] = data

        sobol_numerator[:, :, iparam] = np.var(inner_expectation, axis=2)

    ratio = sobol_numerator / sobol_denominator[:, :, np.newaxis]
    sobol_index = np.mean(ratio, axis=(0, 1))
    print("Total effect sobol index: {}".format(1 - sobol_index))
