import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class sobol_index():
    def __init__(
            self,
            forward_model,
            prior_mean,
            prior_cov,
            model_noise_cov_scalar,
            global_num_outer_samples,
            global_num_inner_samples,
            save_path,
            ytrain=None,
            log_file=None,
            inner_batch_size=1000
            ):

        self.forward_model = forward_model
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.model_noise_cov_scalar = model_noise_cov_scalar
        self.global_num_outer_samples = global_num_outer_samples
        self.global_num_inner_samples = global_num_inner_samples
        self.log_file = log_file
        self.ytrain = ytrain
        self.save_path = save_path

        self.num_parameters = self.prior_mean.shape[0]
        self.num_data_samples, self.spatial_resolution = self.ytrain.shape

        self.local_num_inner_samples = self.global_num_inner_samples
        self.local_num_outer_samples = self.compute_local_num_outer_samples()
        self.inner_batch_size = inner_batch_size

        assert((self.local_num_inner_samples % self.inner_batch_size) ==  0), "Evenly divide inner batch"

    def compute_local_num_outer_samples(self):
        """Function computes the number of local number of samples"""
        assert(self.global_num_outer_samples %
               size == 0), "Equally divide the outer expectation samples"
        return int(self.global_num_outer_samples/size)

    def compute_model_prediction(self, theta):
        """Function computes the model prediction"""
        num_samples = theta.shape[1]
        prediction = np.zeros(self.ytrain.shape+(num_samples, ))
        for isample in range(num_samples):
            prediction[:, :, isample] = self.forward_model(theta[:, isample])
        return prediction

    def sample_gaussian(self, mean, cov, num_samples):
        """Function samples the gaussian"""
        # Definitions
        d, N = mean.shape
        product = d*N*num_samples

        if N > 1:
            assert(np.prod(cov.shape) == 1), "Must provide scalar cov"
            L = np.linalg.cholesky(cov+np.eye(1)*1e-8)
            noise = L@np.random.randn(product).reshape(d, N, num_samples)
            return np.expand_dims(mean, axis=-1) + noise
        else:
            L = np.linalg.cholesky(cov+np.eye(d)*1e-8)
            noise = L@np.random.randn(product).reshape(d, num_samples)
            return mean + noise

    def compute_sobol_indices(self):
        """Function computes the sobol indices"""

        local_outer_expectation = np.zeros_like(self.ytrain)
        sobol_index = np.zeros(self.num_parameters)

        for iparameter in range(self.num_parameters):

            parameter_pair = np.array([iparameter])
            # Selected parameter stats
            selected_mean, selected_cov = self.get_selected_parameter_stats(parameter_pair=parameter_pair)
            # Generate outer samples
            outer_selected_parameter_samples = self.sample_gaussian(
                    mean=selected_mean,
                    cov=selected_cov,
                    num_samples=self.local_num_outer_samples
                    )
            # Compute expectation
            local_inner_expectation_samples = self.compute_inner_expectation(
                    outer_selected_parameter_samples=outer_selected_parameter_samples,
                    parameter_pair=parameter_pair
                    )

            # Reshaping (Assumming data samples are i.i.d.)
            local_reshaped_inner_expectation_samples = local_inner_expectation_samples.reshape(-1, self.local_num_outer_samples)
            local_batch_unnormalized_mean = (self.local_num_outer_samples / self.global_num_outer_samples)*np.mean(local_reshaped_inner_expectation_samples)
            global_batch_unnormalized_mean_list = comm.gather(local_batch_unnormalized_mean, root=0)
            if rank == 0:
                global_mean = [sum(global_batch_unnormalized_mean_list)]*size
            else:
                global_mean = None

            local_mean = comm.scatter(global_mean, root=0)

            local_inner_expectation_samples_mean_removed_sq = (local_inner_expectation_samples - local_mean)**2
            # Reshaping (Assuming data samples are i.i.d.)
            local_inner_expectation_samples_mean_removed_sq_reshaped = local_inner_expectation_samples_mean_removed_sq.reshape(-1, self.local_num_outer_samples)
            local_batch_unnormalized_var = (self.local_num_outer_samples / self.global_num_outer_samples) * np.mean(local_inner_expectation_samples_mean_removed_sq_reshaped)
            global_batch_unnormalized_var_list = comm.gather(local_batch_unnormalized_var, root=0)
            if rank == 0:
                global_var = sum(global_batch_unnormalized_var_list)
                sobol_index[iparameter] = global_var

        self.save_data(data=sobol_index, file_name="sobol_index.npy")
    
    def compute_inner_expectation(self, outer_selected_parameter_samples, parameter_pair):
        """Function computes the inner expectation"""
        local_inner_expectation_samples = np.zeros(self.ytrain.shape+(self.local_num_outer_samples, ))
        inner_sample_parameter = self.get_fixed_parameter_id(parameter_pair=parameter_pair)

        # Selected parameter stats
        selected_mean, selected_cov = self.get_selected_parameter_stats(parameter_pair=inner_sample_parameter)
        # Generate outer samples
        inner_selected_parameter_samples = self.sample_gaussian(
                mean=selected_mean,
                cov=selected_cov,
                num_samples=self.local_num_inner_samples
                )

        eval_parameter = np.zeros((self.num_parameters, self.local_num_inner_samples))
        eval_parameter[inner_sample_parameter, :] = inner_selected_parameter_samples

        for iouter in range(self.local_num_outer_samples):

            # Generate eval samples
            eval_parameter[parameter_pair, :] = outer_selected_parameter_samples[0, iouter]
            
            num_inner_batches = int(self.local_num_inner_samples / self.inner_batch_size)
            assert(num_inner_batches > 0), "Batch size > local inner samples"
            
            inner_expectation = np.zeros_like(self.ytrain)

            for ibatch in range(num_inner_batches):
                inner_batch_samples = eval_parameter[:, ibatch*self.inner_batch_size: (ibatch+1)*self.inner_batch_size]
                inner_batch_prediction = self.compute_model_prediction(theta=inner_batch_samples)
                inner_batch_expectation = np.mean(inner_batch_prediction, axis=-1)
                inner_expectation += (self.inner_batch_size / self.local_num_inner_samples)*inner_batch_expectation

            local_inner_expectation_samples[:, :, iouter] = inner_expectation

        return local_inner_expectation_samples

    def get_fixed_parameter_id(self, parameter_pair):
        """Function returns the idx of fixed parameters given the parameter pair"""
        num_parameter = parameter_pair.shape[0]
        parameter_list = np.arange(self.num_parameters)
        conditions = []
        for ii in range(num_parameter):
            conditions.append(parameter_pair[ii] != parameter_list)
        idx = np.prod(np.array(conditions), axis=0) == 1
        return parameter_list[idx]

    def get_sample_parameter_id(self, parameter_pair):
        """Function returns the idx of sample parameters given the parameter pair"""
        return parameter_pair

    def get_selected_parameter_stats(self, parameter_pair):
        """Function selectes the parameter pair
        Assumptions: parameters are assumed to be uncorrelated (prior to observing the data)"""
        mean = self.prior_mean[parameter_pair, :].reshape(
            parameter_pair.shape[0], 1)
        cov = np.diag(np.diag(self.prior_cov)[parameter_pair])
        return mean, cov

    def save_data(self, data, file_name):
        """Function saves the data"""
        if rank == 0:
            file_path = os.path.join(self.save_path, file_name)
            np.save(file_path, data)

