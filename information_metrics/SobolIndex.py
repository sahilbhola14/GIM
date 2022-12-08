import numpy as np
from mpi4py import MPI

import os
import shutil

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()


class SobolIndex:
    def __init__(
        self,
        forward_model,
        prior_mean,
        prior_cov,
        global_num_outer_samples,
        global_num_inner_samples,
        model_noise_cov_scalar,
        data_shape,
        save_path,
        write_log_file=False,
    ):

        self._forward_model = forward_model
        self._prior_mean = prior_mean
        self._prior_cov = prior_cov
        self._global_num_outer_samples = global_num_outer_samples
        self._global_num_inner_samples = global_num_inner_samples
        self._model_noise_cov_scalar = model_noise_cov_scalar
        self._data_shape = data_shape
        self._save_path = save_path

        self._num_parameters = self._prior_mean.shape[0]

        # Compute worker specific properties
        self._worker_num_outer_samples = self._comp_local_num_outer_samples()
        self._worker_num_inner_samples = self._global_num_inner_samples

        assert (
            self._forward_model(self._prior_mean).shape == self._data_shape
        ), "Data shape does not match"

        # File I/O
        self._create_save_path()
        self._log_file = self._create_log_file(write_log_file)

        self._write_message_to_log_file(
            "Num Outer Samples(Global): {}".format(self._global_num_outer_samples)
        )

        self._write_message_to_log_file(
            "Num Inner Samples(Global): {}".format(self._global_num_inner_samples)
        )

        self._write_message_to_log_file("Num Procs: {}".format(SIZE))

        self._write_message_to_log_file(
            "Num Outer Samples(Worker): {}".format(self._worker_num_outer_samples)
        )
        self._write_message_to_log_file(
            "Num Inner Samples(Worker): {}".format(self._worker_num_inner_samples)
        )

        self._write_message_to_log_file(
            "Number of parameters: {}".format(self._num_parameters)
        )

        self._write_message_to_log_file(
            "Model Noise Cov (scalar): {}".format(self._model_noise_cov_scalar)
        )

    def _create_save_path(self):
        """Creates the save path"""
        if RANK == 0:
            if not os.path.exists(self._save_path):
                os.makedirs(self._save_path)
            else:
                shutil.rmtree(self._save_path)
                os.makedirs(self._save_path)
        else:
            pass

    def _create_log_file(self, write_log_file):
        """Creates the log file
        Returns:
            Log file
        """
        if write_log_file:
            if RANK == 0:
                log_file = open(self._save_path + "/log_SobolIndex.txt", "w")
                log_file.write("Log file (Sobol Index)\n")
                log_file.flush()
            else:
                log_file = None
        else:
            log_file = None

        return log_file

    def _comp_local_num_outer_samples(self):
        """Function computes the number of local number of samples
        Returns:
            (int): Number of worker specific outer samples
        """
        assert (
            self._global_num_outer_samples % SIZE == 0
        ), "Equally divide the outer expectation samples"
        return int(self._global_num_outer_samples / SIZE)

    def _comp_model_prediction(self, theta):
        """Compute the model prediciton

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            theta (float[num_parameters, num_samples]): Parameter sample

        Returns:
            (float): Model prediciton

        """
        assert (
            theta.shape[0] == self._num_parameters
        ), "Number of parameters does not match"
        num_samples = theta.shape[1]
        model_prediction = np.zeros(self._data_shape + (num_samples,))

        for isample in range(num_samples):
            model_prediction[:, :, isample] = self._forward_model(theta[:, isample])

        return model_prediction

    def _sample_model_likelihood(self, theta):
        """Function computes the model likelihood
        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            theta (float[num_parameters, num_samples]): Parameter sample

        Returns:
            (float[num_samples, 1]): Model likelihood samples

        """
        assert theta.shape[0] == self._num_parameters, "Incorrect number of parameters"

        num_samples = theta.shape[1]
        noise_sample = np.sqrt(self._model_noise_cov_scalar) * np.random.randn(
            self._data_shape[0], self._data_shape[1], num_samples
        )
        model_prediction = self._comp_model_prediction(theta)
        likelihood_sample = model_prediction + noise_sample

        return likelihood_sample

    def _sample_gaussian(self, mean, cov, num_samples):
        """Sample from a Gaussian distribution

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            mean (float[num_parameters, 1]): Mean of the Gaussian distribution
            cov (float[num_parameters, num_parameters]): Covariance of the Gaussian
            distribution
            num_samples (int): Number of samples

        Returns:
            (float[num_parameters, num_samples]): Samples from the Gaussian distribution

        """
        assert mean.shape[0] == cov.shape[0], "Number of parameters does not match"
        assert mean.shape[1] == 1, "Input mean should be a column vector"
        assert num_samples > 0, "Number of samples must be greater than zero"

        return np.random.multivariate_normal(mean.ravel(), cov, num_samples).T

    def _comp_selected_parameter_samples(self, parameter_id, num_samples):
        """Computes the selected parameter samples

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            parameter_id (int): Parameter id for which samples are computed
            num_samples (int): Number of samples

        Returns:
            (float): Samples of the selected parameter
        """
        mean = self._prior_mean[parameter_id, 0].reshape(-1, 1)
        cov = self._prior_cov[parameter_id, :][:, parameter_id]
        return self._sample_gaussian(mean, cov, num_samples)

    def _comp_sobol_denominator_via_samples(self):
        """Computes the Sobol denominator via Samples
        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Returns:
            (float): Sobol denominator
        """
        # Compute outer samples
        outer_parameter_samples = self._comp_selected_parameter_samples(
            parameter_id=np.arange(self._num_parameters),
            num_samples=self._worker_num_outer_samples,
        )

        # Compute model prediction
        model_likelihood = self._sample_model_likelihood(outer_parameter_samples)

        # Compute Sobol denominator
        unnormalized_mean_model_prediction = self._worker_num_outer_samples * np.mean(
            model_likelihood, axis=2
        )

        unnormalized_mean_model_prediction_list = COMM.gather(
            unnormalized_mean_model_prediction, root=0
        )

        if RANK == 0:
            global_mean_model_prediction = (
                1 / self._global_num_outer_samples
            ) * np.sum(unnormalized_mean_model_prediction_list, axis=0)
            global_mean_model_prediction_list = [global_mean_model_prediction] * SIZE
        else:
            global_mean_model_prediction_list = None

        COMM.Barrier()
        local_mean_model_prediction = COMM.scatter(
            global_mean_model_prediction_list, root=0
        )

        error = model_likelihood - local_mean_model_prediction[:, :, np.newaxis]
        unnormalized_mean_squared_error = self._worker_num_outer_samples * np.mean(
            error**2, axis=2
        )
        unnormalized_mean_squation_error_list = COMM.gather(
            unnormalized_mean_squared_error, root=0
        )

        if RANK == 0:
            global_mean_squared_error = (1 / self._global_num_outer_samples) * np.sum(
                unnormalized_mean_squation_error_list, axis=0
            )
            global_mean_squared_error_list = [global_mean_squared_error] * SIZE
        else:
            global_mean_squared_error_list = None

        sobol_denominator = COMM.scatter(global_mean_squared_error_list, root=0)

        assert (
            sobol_denominator.shape == self._data_shape
        ), "Sobol denominator shape does not match"
        assert (sobol_denominator > 0).all(), "Sobol denominator should be positive"
        assert (sobol_denominator < np.inf).all(), "Sobol denominator should be finite"

        return sobol_denominator

    def _comp_inner_expectation(
        self, selected_parameter_id, outer_selected_parameter_samples
    ):
        """Computes the inner expectation

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            selected_parameter_id (int): Parameter id for outer samples
            outer_selected_parameter_samples (float): Samples of the selected parameter

        Returns:
            (float): Inner expectation
        """
        inner_expectation = np.zeros(
            self._data_shape + (self._worker_num_outer_samples,)
        )
        unselected_parameter_id = np.delete(
            np.arange(self._num_parameters), selected_parameter_id
        )

        # Compute inner samples
        inner_unselected_parameter_samples = self._comp_selected_parameter_samples(
            parameter_id=unselected_parameter_id,
            num_samples=self._worker_num_inner_samples,
        )

        for iouter in range(self._worker_num_outer_samples):

            if iouter % 100 == 0 or iouter == self._worker_num_outer_samples - 1:
                self._write_message_to_log_file(
                    " > Computing inner expectation for outer sample {}/{}".format(
                        iouter + 1, self._worker_num_outer_samples
                    )
                )

            # Compute model prediction
            theta = np.zeros((self._num_parameters, self._worker_num_inner_samples))
            theta[selected_parameter_id, :] = outer_selected_parameter_samples[
                :, iouter
            ]
            theta[unselected_parameter_id, :] = inner_unselected_parameter_samples
            model_likelihood = self._sample_model_likelihood(theta)

            # Compute inner expectation
            inner_expectation[:, :, iouter] = np.mean(model_likelihood, axis=2)

        return inner_expectation

    def _write_message_to_log_file(self, message, write_rank=0):
        """Writes a message to the log file
        Args:
            message (str): Message to be written
        """
        if RANK == write_rank:
            if self._log_file is None:
                print(message + "\n", flush=True)
            else:
                self._log_file.write(message + "\n")
                self._log_file.flush()

    def _save_data(self, data, filename):
        """Saves data to a file
        Args:
            data (float): Data to be saved
            filename (str): Filename
        """
        if RANK == 0:
            file_path = os.path.join(self._save_path, filename)
            np.save(file_path, data)
        else:
            pass

    def _comp_variance_of_inner_expectation(self, inner_expectation):
        """Function to compute the variance of the inner Expectation
        Args:
            inner_expectation (float): Inner Expectation samples

        Returns:
            (float): Variance of the inner Expectation
        """
        unnormalized_mean_inner_expectation = self._worker_num_outer_samples * np.mean(
            inner_expectation, axis=2
        )

        unnormalized_mean_inner_expectation_list = COMM.gather(
            unnormalized_mean_inner_expectation, root=0
        )

        if RANK == 0:
            global_mean_inner_expectation = (
                1 / self._global_num_outer_samples
            ) * np.sum(unnormalized_mean_inner_expectation_list, axis=0)
            global_mean_inner_expectation_list = [global_mean_inner_expectation] * SIZE
        else:
            global_mean_inner_expectation_list = None

        COMM.Barrier()
        local_mean_inner_expectation = COMM.scatter(
            global_mean_inner_expectation_list, root=0
        )

        error = inner_expectation - local_mean_inner_expectation[:, :, np.newaxis]
        unnormalized_mean_squared_error = self._worker_num_outer_samples * np.mean(
            error**2, axis=2
        )
        unnormalized_mean_squation_error_list = COMM.gather(
            unnormalized_mean_squared_error, root=0
        )

        if RANK == 0:
            global_mean_squared_error = (1 / self._global_num_outer_samples) * np.sum(
                unnormalized_mean_squation_error_list, axis=0
            )
            global_mean_squared_error_list = [global_mean_squared_error] * SIZE
        else:
            global_mean_squared_error_list = None

        COMM.Barrier()

        variance_of_inner_expectation = COMM.scatter(
            global_mean_squared_error_list, root=0
        )

        return variance_of_inner_expectation

    def comp_sobol_indices(self):
        """Compute the Sobol index

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Returns:
            (float[num_parameters, 1]): Sobol index

        """
        sobol_numerator = np.zeros(self._data_shape + (self._num_parameters,))
        sobol_denominator = np.zeros(self._data_shape)

        # Compute the Sobol Numerator
        for iparam in range(self._num_parameters):

            self._write_message_to_log_file(
                message="Computing Sobol numerator for parameter {}/{}".format(
                    iparam + 1, self._num_parameters
                )
            )

            selected_parameter_id = np.array([iparam])

            # Compute outer samples
            outer_selected_parameter_samples = self._comp_selected_parameter_samples(
                parameter_id=selected_parameter_id,
                num_samples=self._worker_num_outer_samples,
            )
            # Compute inner expectation for each outer selected parameter sample
            inner_expectation = self._comp_inner_expectation(
                selected_parameter_id=selected_parameter_id,
                outer_selected_parameter_samples=outer_selected_parameter_samples,
            )

            # Compute Variance of Inner Expectation
            sobol_numerator[:, :, iparam] = self._comp_variance_of_inner_expectation(
                inner_expectation
            )

        # Compute the Sobol Denominator
        sobol_denominator = self._comp_sobol_denominator_via_samples()

        # Compute the Sobol Index
        ratio = sobol_numerator / sobol_denominator[:, :, np.newaxis]
        sobol_index = np.mean(ratio, axis=(0, 1))

        self._write_message_to_log_file(message="Sobol index computation completed!")
        self._write_message_to_log_file(message="Sobol index: {}".format(sobol_index))

        # Save the data
        self._save_data(sobol_index, filename="sobol_index.npy")
