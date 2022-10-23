import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import os
import shutil

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
            inner_batch_size=500,
            restart=False,
            inner_expectation_save_restart_freq=50
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
        self.restart = restart

        self.num_parameters = self.prior_mean.shape[0]
        self.num_data_samples, self.spatial_resolution = self.ytrain.shape

        self.local_num_inner_samples = self.global_num_inner_samples
        self.local_num_outer_samples = self.compute_local_num_outer_samples()
        self.inner_batch_size = inner_batch_size
        self.inner_expectation_save_restart_freq = inner_expectation_save_restart_freq

        self.restart_file_path = os.path.join(self.save_path, "restart_files_sobol")
        if self.restart is False:
            if rank == 0:
                self.create_restart_folders()
                self.remove_file("sobol_index.npy")
        comm.Barrier()


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
        
            self.write_log_file( ">>>-----------------------------------------------------------<<<")
            self.write_log_file("Computing Sobol index for Theta : {} / {}".format(iparameter+1, self.num_parameters))
            self.write_log_file( ">>>-----------------------------------------------------------<<<")

            parameter_pair = np.array([iparameter])

            # Try loading the outer prior samples
            outer_selected_parameter_samples, is_outer_selected_parameter_samples_avail = self.load_restart_data(
                    data_type="outer",
                    label="outer_prior_samples_parameter_{}".format(iparameter)
                    )

            if is_outer_selected_parameter_samples_avail is False:

                # Selected parameter stats
                selected_mean, selected_cov = self.get_selected_parameter_stats(parameter_pair=parameter_pair)

                # Generate outer samples
                outer_selected_parameter_samples = self.sample_gaussian(
                        mean=selected_mean,
                        cov=selected_cov,
                        num_samples=self.local_num_outer_samples
                    )

                # Save the prior samples
                self.save_restart_data(
                        data=outer_selected_parameter_samples,
                        data_type="outer",
                        label="outer_prior_samples_parameter_{}".format(iparameter)
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
        self.write_log_file("\nDone computing sobol index for all parameters! Save path : ./sobol_index.npy") 
    
    def compute_inner_expectation(self, outer_selected_parameter_samples, parameter_pair):
        """Function computes the inner expectation"""
        # local inner files
        inner_expectation_counter_file_name = "inner_expectation_counter_parameter_{}".format(parameter_pair[0])
        inner_expectation_samples_file_name = "inner_expectation_samples_parameter_{}".format(parameter_pair[0])

        # Load data
        local_inner_expectation_samples, is_local_inner_expectation_samples_avail = self.load_restart_data(
                data_type="inner",
                label=inner_expectation_samples_file_name
                )

        # Load counter
        local_inner_expectation_counter, is_local_inner_expectation_counter_avail = self.load_restart_data(
                data_type="inner",
                label=inner_expectation_counter_file_name
                )
        if is_local_inner_expectation_samples_avail is False:
            local_inner_expectation_samples = np.nan*np.ones(self.ytrain.shape+(self.local_num_outer_samples, ))

        if is_local_inner_expectation_counter_avail is False:
            local_inner_expectation_counter = 0
            local_lower_loop_idx = 0
        else:
            local_lower_loop_idx = local_inner_expectation_counter.item()
            self.write_log_file("Inner expectation restarted from : {}/{}".format(local_lower_loop_idx, self.local_num_outer_samples))

        # Pending evaluations
        num_pending_inner_evaluations = self.local_num_outer_samples - local_inner_expectation_counter

        # Inner sample parameter
        inner_sample_parameter = self.get_fixed_parameter_id(parameter_pair=parameter_pair)

        # Selected parameter stats
        selected_mean, selected_cov = self.get_selected_parameter_stats(parameter_pair=inner_sample_parameter)

        # Generate inner samples
        inner_selected_parameter_samples = self.sample_gaussian(
                mean=selected_mean,
                cov=selected_cov,
                num_samples=self.local_num_inner_samples
                )

        eval_parameter = np.zeros((self.num_parameters, self.local_num_inner_samples))
        eval_parameter[inner_sample_parameter, :] = inner_selected_parameter_samples

        for iouter in range(local_lower_loop_idx, self.local_num_outer_samples):
            if (iouter%50 == 0) or (iouter==self.local_num_outer_samples-1):
                self.write_log_file("     (Inner expectation) Outer sample #{}/{}".format(iouter+1, self.local_num_outer_samples))

            # Build eval samples
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

            local_inner_expectation_counter += 1

            save_condition = iouter % self.inner_expectation_save_restart_freq

            if (save_condition == 0) or (iouter == self.local_num_outer_samples - 1):
                
                # Saving the inner expectation samples
                self.save_restart_data(
                        data=local_inner_expectation_samples,
                        data_type="inner",
                        label=inner_expectation_samples_file_name
                        )

                self.save_restart_data(
                        data=local_inner_expectation_counter,
                        data_type="inner",
                        label=inner_expectation_counter_file_name
                        )

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

    def write_log_file(self, message):
        """Function writes the log file"""
        if rank == 0:
            self.log_file.write(message+"\n")
            self.log_file.flush()
        else:
            pass

    def create_restart_folders(self):
        """Function creates restart folder"""
        if os.path.exists(self.restart_file_path):
            shutil.rmtree(self.restart_file_path)
        os.mkdir(self.restart_file_path)
        os.mkdir(os.path.join(self.restart_file_path, "outer_data"))
        os.mkdir(os.path.join(self.restart_file_path, "inner_data"))

    def remove_file(self, file_name):
        """Function removes the files"""
        file_name = os.path.join(self.save_path, file_name)
        if rank == 0:
            if os.path.exists(file_name):
                os.remove(file_name)
        else:
            pass

    def load_restart_data(self, data_type, label):
        """Function loads the data"""
        if data_type == "outer":
            folder = "outer_data"
        elif data_type == "inner":
            folder = "inner_data"

        file_path = os.path.join( self.restart_file_path, folder, label+"_rank_{}.npy".format(rank))

        file_exists = os.path.exists(file_path)
        if file_exists:
            return np.load(file_path), file_exists
        else:
            return np.nan, file_exists

    def save_restart_data(self, data, data_type, label):
        """Function saves the restart data"""
        if data_type == "outer":
            folder = "outer_data"
        else:
            folder = "inner_data"

        file_path = os.path.join(
            self.restart_file_path, folder, label+"_rank_{}.npy".format(rank))

        np.save(file_path, data)




