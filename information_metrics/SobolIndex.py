import numpy as np
from mpi4py import MPI

# import os
# import shutil

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
        data_shape,
    ):

        self._forward_model = forward_model
        self._prior_mean = prior_mean
        self._prior_cov = prior_cov
        self._global_num_outer_samples = global_num_outer_samples
        self._global_num_inner_samples = global_num_inner_samples
        self._data_shape = data_shape

        self._num_parameters = self._prior_mean.shape[0]

        # Compute worker specific properties
        self._worker_num_outer_samples = self._comp_local_num_outer_samples()
        self._worker_num_inner_samples = self._global_num_inner_samples

        assert (
            self._forward_model(self._prior_mean).shape == self._data_shape
        ), "Data shape does not match"

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

    def _sample_gaussian(self, mean, cov, num_samples):
        """Sample from a Gaussian distribution

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            mean (float[num_parameters, 1]): Mean of the Gaussian distribution
            cov (float[num_parameters, num_parameters]): Covariance of the Gaussian distribution
            num_samples (int): Number of samples

        Returns:
            (float[num_parameters, num_samples]): Samples from the Gaussian distribution

        """
        assert mean.shape[0] == cov.shape[0], "Number of parameters does not match"
        assert mean.shape[1] == 1, "Input mean should be a column vector"
        assert num_samples > 0, "Number of samples must be greater than zero"

        return np.random.multivariate_normal(mean.ravel(), cov, num_samples).T

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
            print("Computing Sobol Numerator for parameter {}".format(iparam))
            selected_parameter_id = np.array([iparam])
            # unselected_parameter_id = np.delete(np.arange(self._num_parameters), iparam)

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

            sobol_numerator[:, :, iparam] = np.var(inner_expectation, axis=2)

        # Compute the Sobol Denominator
        sobol_denominator = self._comp_sobol_denominator_via_samples()

        # Compute the Sobol Index
        sobol_index = sobol_numerator / sobol_denominator[:, :, np.newaxis]
        print("Sobol Index: ", np.mean(sobol_index, axis=(0, 1)))
        print(np.sum(np.mean(sobol_index, axis=(0, 1))))

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

    def _comp_inner_expectation(
        self, selected_parameter_id, outer_selected_parameter_samples
    ):
        """Computes the inner expectation

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            selected_parameter_id (int): Parameter id for which outer samples are computed
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
            print("Computing inner expectation for outer sample {}".format(iouter))
            # Compute model prediction
            theta = np.zeros((self._num_parameters, self._worker_num_inner_samples))
            theta[selected_parameter_id, :] = outer_selected_parameter_samples[
                :, iouter
            ]
            theta[unselected_parameter_id, :] = inner_unselected_parameter_samples
            model_prediction = self._comp_model_prediction(theta)

            # Compute inner expectation
            inner_expectation[:, :, iouter] = np.mean(model_prediction, axis=2)

        return inner_expectation

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
        model_prediction = self._comp_model_prediction(outer_parameter_samples)

        # Compute Sobol denominator
        sobol_denominator = np.var(model_prediction, axis=2)
        assert (
            sobol_denominator.shape == self._data_shape
        ), "Sobol denominator shape does not match"
        assert (sobol_denominator > 0).all(), "Sobol denominator should be positive"
        assert (sobol_denominator < np.inf).all(), "Sobol denominator should be finite"
        return sobol_denominator
