# Author: Sahil Bhola (University of Michigan)
# Date: 14 June 2022
# Description : Computes the model identifiability using Mutual information 

import numpy as np
from mpi4py import MPI
import sys
from itertools import combinations
import os
import shutil
import matplotlib.pyplot as plt
sys.path.append("/home/sbhola/Documents/CASLAB/GIM/quadrature")
from quadrature import unscented_quadrature, gauss_hermite_quadrature

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class mutual_information():
    def __init__(
            self,
            forward_model,
            prior_mean,
            prior_cov,
            model_noise_cov_scalar,
            global_num_outer_samples,
            global_num_inner_samples,
            save_path,
            restart,
            ytrain=None,
            log_file=None,
            mutual_information_evidence_save_restart_freq=50,
            conditional_mutual_information_save_restart_freq=50
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
        self.mutual_information_evidence_save_restart_freq = mutual_information_evidence_save_restart_freq
        self.conditional_mutual_information_save_restart_freq = conditional_mutual_information_save_restart_freq
        self.restart = restart

        self.num_parameters = self.prior_mean.shape[0]

        self.local_num_inner_samples = self.global_num_inner_samples
        self.local_num_outer_samples = self.compute_local_num_outer_samples()

        self.local_outer_prior_samples = np.zeros(
            (self.num_parameters, self.local_num_outer_samples))

        self.local_outer_model_prediction = np.zeros(
            self.ytrain.shape+(self.local_num_outer_samples, ))

        self.local_outer_data_samples = np.zeros(
            self.local_outer_model_prediction.shape)

        self.local_outer_likelihood_prob = np.zeros(
            self.local_num_outer_samples)

        self.outer_data_computation_req_flag = True

        self.num_data_samples = self.ytrain.shape[0]
        self.spatial_res = self.ytrain.shape[1]

        self.restart_file_path = os.path.join(self.save_path, "restart_files")
        if self.restart is False:
            if rank == 0:
                self.create_restart_folders()
                self.remove_file("estimated_mutual_information.npy")
                self.remove_file("estimated_individual_mutual_information.npy")
                self.remove_file("estimated_pair_mutual_information.npy")
        comm.Barrier()

    def compute_local_num_outer_samples(self):
        """Function computes the number of local number of samples"""
        assert(self.global_num_outer_samples %
               size == 0), "Equally divide the outer expectation samples"
        return int(self.global_num_outer_samples/size)

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

    def sample_parameter_distribution(self, num_samples, parameter_pair=None):
        """Function samples the parameter distribution"""
        if parameter_pair is None:
            # parameter_samples = self.sample_gaussian(
            #     mean=self.prior_mean,
            #     cov=self.prior_cov,
            #     num_samples=num_samples
            # )
            raise ValueError("Must provide parameter pair")
        else:
            selected_parameter_mean, selected_parameter_cov = self.get_selected_parameter_stats(
                parameter_pair=parameter_pair)
            parameter_samples = self.sample_gaussian(
                mean=selected_parameter_mean,
                cov=selected_parameter_cov,
                num_samples=num_samples
            )

        return parameter_samples

    def sample_prior_distribution(self, num_samples):
        """Function samples the prior distribution"""
        prior_samples = self.sample_gaussian(
            mean=self.prior_mean,
            cov=self.prior_cov,
            num_samples=num_samples
        )
        return prior_samples

    def get_selected_parameter_stats(self, parameter_pair):
        """Function selectes the parameter pair
        Assumptions: parameters are assumed to be uncorrelated (prior to observing the data)"""
        mean = self.prior_mean[parameter_pair, :].reshape(
            parameter_pair.shape[0], 1)
        cov = np.diag(np.diag(self.prior_cov)[parameter_pair])
        return mean, cov

    def compute_model_prediction(self, theta):
        """Function computes the model prediction"""
        num_samples = theta.shape[1]
        prediction = np.zeros(self.ytrain.shape+(num_samples, ))
        for isample in range(num_samples):
            prediction[:, :, isample] = self.forward_model(theta[:, isample])
        return prediction

    def sample_likelihood(self, theta):
        """Function samples the likelihood"""
        model_prediction = self.compute_model_prediction(theta=theta)
        product = np.prod(model_prediction.shape)
        noise = np.sqrt(self.model_noise_cov_scalar) * \
            np.random.randn(product).reshape(model_prediction.shape)
        likelihood_sample = model_prediction + noise
        return likelihood_sample, model_prediction

    def evaluate_likelihood_probaility(self, data, model_prediction):
        """Function evaluates the likelihood probability"""
        error = data - model_prediction
        error_norm_sq = np.linalg.norm(error, axis=1)**2
        pre_exp = 1 / ((2*np.pi*self.model_noise_cov_scalar)
                       ** (self.spatial_res/2))

        likelihood = (pre_exp**self.num_data_samples) * \
            np.exp(-0.5*np.sum(error_norm_sq / self.model_noise_cov_scalar, axis=0))
        return likelihood

    def estimate_mutual_information_via_mc(self, use_quadrature=False, quadrature_rule="gaussian", num_gaussian_quad_pts=50):
        """Function computes the mutual information via mc"""
        self.write_log_file(
            ">>>-----------------------------------------------------------<<<")
        self.write_log_file(
            ">>> Begin computing the mutual information, I(theta;Y) <<<")
        self.write_log_file(
            ">>>-----------------------------------------------------------<<<\n")
        if self.outer_data_computation_req_flag is True:
            # Generate the prior samples ~ p(\theta)
            self.local_outer_prior_samples, is_local_outer_prior_samp_avail = self.load_restart_data(
                data_type="outer",
                label="outer_prior_samples"
            )
            if is_local_outer_prior_samp_avail is False:
                self.local_outer_prior_samples = self.sample_prior_distribution(
                    num_samples=self.local_num_outer_samples)

            # Generate data samples from the conditional distribution, p(y | \theta)
            self.local_outer_data_samples, is_local_outer_data_samp_avail = self.load_restart_data(
                data_type="outer",
                label="outer_data_samples"
            )
            if is_local_outer_data_samp_avail is False:
                self.local_outer_data_samples, self.local_outer_model_prediction = self.sample_likelihood(
                    theta=self.local_outer_prior_samples)

            # Likelihood probability
            self.local_outer_likelihood_prob, is_local_outer_likelihood_prob_avail = self.load_restart_data(
                data_type="outer",
                label="outer_likelihood_prob"
            )

            if is_local_outer_likelihood_prob_avail is False:
                self.local_outer_likelihood_prob = self.evaluate_likelihood_probaility(
                    data=self.local_outer_data_samples,
                    model_prediction=self.local_outer_model_prediction
                )

            self.save_restart_data(
                data=self.local_outer_prior_samples,
                data_type="outer",
                label="outer_prior_samples"
            )
            self.save_restart_data(
                data=self.local_outer_data_samples,
                data_type="outer",
                label="outer_data_samples"
            )

            self.save_restart_data(
                data=self.local_outer_likelihood_prob,
                data_type="outer",
                label="outer_likelihood_prob"
            )

            self.outer_data_computation_req_flag = False

        local_log_likelihood = np.log(self.local_outer_likelihood_prob)

        # Compute evidence
        local_evidence_prob = self.estimate_evidence_probability(
            use_quadrature=use_quadrature,
            quadrature_rule=quadrature_rule,
            num_gaussian_quad_pts=num_gaussian_quad_pts
        )
        local_log_evidence = np.log(local_evidence_prob)

        comm.Barrier()
        global_log_likelihood = comm.gather(local_log_likelihood, root=0)
        global_log_evidence = comm.gather(local_log_evidence, root=0)

        if rank == 0:
            summation = sum([np.sum(global_log_likelihood[ii] -
                            global_log_evidence[ii]) for ii in range(size)])
            mutual_information = (1/self.global_num_outer_samples)*summation

            self.save_quantity(
                "estimated_mutual_information.npy", mutual_information)
            self.write_log_file(
                ">>> End computing the mutual information, I(theta;Y) <<<")

    def estimate_evidence_probability(self, num_gaussian_quad_pts, use_quadrature=False, quadrature_rule="gaussian"):
        """Function estimates the evidence probability"""
        if use_quadrature is True:
            evidence_prob = self.integrate_likelihood_via_quadrature(
                quadrature_rule=quadrature_rule,
                num_gaussian_quad_pts=num_gaussian_quad_pts)
        else:
            evidence_prob = self.integrate_likelihood_via_mc()

        return evidence_prob

    def integrate_likelihood_via_mc(self):
        """Function integrates the likelihood to estimate the evidence"""
        # Definitions
        evidence_prob, is_evidence_prob_avail = self.load_restart_data(
            data_type="inner",
            label="evidence_prob",
            sub_type="MI"
        )

        local_mutual_information_inner_counter, is_mi_inner_counter_avail = self.load_restart_data(
            data_type="inner",
            label="mutual_infomation_inner_counter",
            sub_type="MI"
        )

        if is_evidence_prob_avail is False:
            evidence_prob = np.nan*np.ones(self.local_num_outer_samples)
        if is_mi_inner_counter_avail is False:
            local_mutual_information_inner_counter = 0
            local_lower_loop_idx = 0
        else:
            local_lower_loop_idx = local_mutual_information_inner_counter.item()

        # Pre computations
        pre_exp = 1 / ((2*np.pi*self.model_noise_cov_scalar)
                       ** (self.spatial_res/2))

        # Compute inner prior samples
        inner_prior_samples = self.sample_prior_distribution(
            num_samples=self.local_num_inner_samples)

        # Compute model prediction
        inner_model_prediction = self.compute_model_prediction(
            theta=inner_prior_samples)

        # Comptute the evidence
        for isample in range(local_lower_loop_idx, self.local_num_outer_samples):
            outer_sample = np.expand_dims(
                self.local_outer_data_samples[:, :, isample], axis=-1)
            error = outer_sample - inner_model_prediction
            error_norm_sq = np.linalg.norm(error, axis=1)**2
            sample_evidence_estimates = pre_exp * \
                np.exp(-0.5*np.sum(error_norm_sq /
                                   self.model_noise_cov_scalar, axis=0))
            evidence_prob[isample] = (
                1/self.local_num_inner_samples)*np.sum(sample_evidence_estimates)

            local_mutual_information_inner_counter += 1

            save_condition = isample % self.mutual_information_evidence_save_restart_freq
            final_save_condition = self.local_num_outer_samples % self.mutual_information_evidence_save_restart_freq

            if save_condition or final_save_condition == 0:
                self.save_restart_data(
                    data=local_mutual_information_inner_counter,
                    data_type="inner",
                    label="mutual_infomation_inner_counter",
                    sub_type="MI"
                )

                self.save_restart_data(
                    data=evidence_prob,
                    data_type="inner",
                    label="evidence_prob",
                    sub_type="MI"
                )

        return evidence_prob

    def integrate_likelihood_via_quadrature(self, quadrature_rule, num_gaussian_quad_pts):
        """Function integrates the likelihood to estimate the evidence using quadratures"""
        # Definitions
        evidence_prob, is_evidence_prob_avail = self.load_restart_data(
            data_type="inner",
            label="evidence_prob",
            sub_type="MI"
        )

        local_mutual_information_inner_counter, is_mi_inner_counter_avail = self.load_restart_data(
            data_type="inner",
            label="mutual_infomation_inner_counter",
            sub_type="MI"
        )

        if is_evidence_prob_avail is False:
            evidence_prob = np.nan*np.ones(self.local_num_outer_samples)
        if is_mi_inner_counter_avail is False:
            local_mutual_information_inner_counter = 0
            local_lower_loop_idx = 0
        else:
            local_lower_loop_idx = local_mutual_information_inner_counter.item()

        for isample in range(local_lower_loop_idx, self.local_num_outer_samples):
            # Extract outer sample
            outer_sample = np.expand_dims(
                self.local_outer_data_samples[:, :, isample], axis=-1)

            # Integrand
            def integrand(eval_theta):
                integrand_val = self.likelihood_integrand(
                    theta=eval_theta,
                    outer_sample=outer_sample
                )
                return integrand_val

            if quadrature_rule == "unscented":
                unscented_quad = unscented_quadrature(
                    mean=self.prior_mean,
                    cov=self.prior_cov,
                    integrand=integrand
                )

                evidence_mean, evidence_cov = unscented_quad.compute_integeral()

                evidence_prob[isample] = evidence_mean + \
                    np.sqrt(evidence_cov)*np.random.randn(1)

            elif quadrature_rule == "gaussian":
                gh = gauss_hermite_quadrature(
                    mean=self.prior_mean,
                    cov=self.prior_cov,
                    integrand=integrand,
                    num_points=num_gaussian_quad_pts
                )
                evidence_prob[isample] = gh.compute_integeral()

            else:
                raise ValueError("Invalid quadrature rule")

            local_mutual_information_inner_counter += 1

            save_condition = isample % self.mutual_information_evidence_save_restart_freq
            final_save_condition = self.local_num_outer_samples % self.mutual_information_evidence_save_restart_freq

            if save_condition or final_save_condition == 0:
                self.save_restart_data(
                    data=local_mutual_information_inner_counter,
                    data_type="inner",
                    label="mutual_infomation_inner_counter",
                    sub_type="MI"
                )

                self.save_restart_data(
                    data=evidence_prob,
                    data_type="inner",
                    label="evidence_prob",
                    sub_type="MI"
                )
        return evidence_prob

    def likelihood_integrand(self, theta, outer_sample):
        """Function returns the integrand evaluated at the quadrature points
        NOTE: shape of the model prediction should be :(num_data_samples, spatial_res, num_samples)"""
        model_prediction = self.compute_model_prediction(theta=theta)
        pre_exp = 1 / ((2*np.pi*self.model_noise_cov_scalar)
                       ** (self.spatial_res/2))

        error = outer_sample - model_prediction
        error_norm_sq = np.linalg.norm(error, axis=1)**2
        sample_evidence_estimates = pre_exp * \
            np.exp(-0.5*np.sum(error_norm_sq /
                               self.model_noise_cov_scalar, axis=0))
        return sample_evidence_estimates.reshape(1, -1)

    def display_messsage(self, message, print_rank="root"):
        """Print function"""
        if print_rank == "root":
            if rank == 0:
                print(message)
        elif print_rank == "all":
            print(message+" at rank : {}".format(rank))
        elif rank == print_rank:
            print(message+" at rank : {}".format(rank))

    def save_quantity(self, file_name, data):
        """Function saves the data"""
        file_name = os.path.join(self.save_path, file_name)
        if rank == 0:
            np.save(os.path.join(self.save_path, file_name), data)
        else:
            pass

    def write_log_file(self, message):
        """Function writes the message on the log file"""
        if rank == 0:
            assert(self.log_file is not None), "log file must be provided"
            self.log_file.write(message+"\n")
            self.log_file.flush()

    def save_restart_data(self, data, data_type, label, sub_type=None):
        """Function saves the data"""
        if data_type == "outer":
            folder = "outer_data"
            file_path = os.path.join(
                self.restart_file_path, folder, label+"_rank_{}.npy".format(rank))
        else:
            folder = "inner_data"

            if sub_type == "MI":
                sub_folder = "mutual_information"
            elif sub_type == "ICMI":
                sub_folder = "individual_conditional_mutual_information"
            elif sub_type == "PCMI":
                sub_folder = "pair_conditional_mutual_information"
            else:
                raise ValueError("Invalid subfolder type")

            file_path = os.path.join(
                self.restart_file_path, folder, sub_folder, label+"_rank_{}.npy".format(rank))

        np.save(file_path, data)

    def load_restart_data(self, data_type, label, sub_type=None):
        """Function loads the data"""
        if data_type == "outer":
            folder = "outer_data"
            file_path = os.path.join(
                self.restart_file_path, folder, label+"_rank_{}.npy".format(rank))
        else:
            folder = "inner_data"

            if sub_type == "MI":
                sub_folder = "mutual_information"
            elif sub_type == "ICMI":
                sub_folder = "individual_conditional_mutual_information"
            elif sub_type == "PCMI":
                sub_folder = "pair_conditional_mutual_information"

            file_path = os.path.join(
                self.restart_file_path, folder, sub_folder, label+"_rank_{}.npy".format(rank))

        file_exists = os.path.exists(file_path)
        if file_exists:
            return np.load(file_path), file_exists
        else:
            return np.nan, file_exists

    def create_restart_folders(self):
        """Function creates the restart folder """
        if os.path.exists(self.restart_file_path):
            shutil.rmtree(self.restart_file_path)
        os.mkdir(self.restart_file_path)
        os.mkdir(os.path.join(self.restart_file_path, "outer_data"))
        os.mkdir(os.path.join(self.restart_file_path, "inner_data"))
        os.mkdir(os.path.join(self.restart_file_path,
                 "inner_data", "mutual_information"))
        os.mkdir(os.path.join(self.restart_file_path, "inner_data",
                 "individual_conditional_mutual_information"))
        os.mkdir(os.path.join(self.restart_file_path, "inner_data",
                 "pair_conditional_mutual_information"))

    def remove_file(self, file_name):
        """Function deletes a file"""
        file_name = os.path.join(self.save_path, file_name)
        if rank == 0:
            if os.path.exists(file_name):
                os.remove(file_name)
        else:
            pass


class conditional_mutual_information(mutual_information):
    def compute_individual_parameter_data_mutual_information_via_mc(self, use_quadrature=False, single_integral_gaussian_quad_pts=60):
        """Function computes the mutual information between each parameter theta_i and data Y given rest of the parameters"""
        # Definitions
        if os.path.exists("estimated_individual_mutual_information.npy"):
            inidividual_mutual_information = np.load(
                "estimated_individual_mutual_information.npy")
        else:
            inidividual_mutual_information = np.nan * \
                np.ones(self.num_parameters)

        self.write_log_file(
            ">>>-----------------------------------------------------------<<<")
        self.write_log_file(
            ">>> Begin computing the individual parameter mutual information, I(theta_i;Y|theta_[rest of parameters])")
        self.write_log_file(
            ">>>-----------------------------------------------------------<<<\n")

        if self.outer_data_computation_req_flag is True:
            # Generate the prior samples ~ p(\theta)
            self.local_outer_prior_samples, is_local_outer_prior_samp_avail = self.load_restart_data(
                data_type="outer",
                label="outer_prior_samples"
            )
            if is_local_outer_prior_samp_avail is False:
                self.local_outer_prior_samples = self.sample_prior_distribution(
                    num_samples=self.local_num_outer_samples)
            else:
                self.write_log_file("Loaded outer prior samples")
                self.write_log_file("----------------")

            # Generate data samples from the conditional distribution, p(y | \theta)
            self.local_outer_data_samples, is_local_outer_data_samp_avail = self.load_restart_data(
                data_type="outer",
                label="outer_data_samples"
            )
            if is_local_outer_data_samp_avail is False:
                self.local_outer_data_samples, self.local_outer_model_prediction = self.sample_likelihood(
                    theta=self.local_outer_prior_samples)

                self.write_log_file("Done computing outer data samples")
                self.write_log_file("----------------")
            else:
                self.write_log_file("Loaded outer data samples")
                self.write_log_file("----------------")

            # Likelihood probability
            self.local_outer_likelihood_prob, is_local_outer_likelihood_prob_avail = self.load_restart_data(
                data_type="outer",
                label="outer_likelihood_prob"
            )

            if is_local_outer_likelihood_prob_avail is False:
                self.local_outer_likelihood_prob = self.evaluate_likelihood_probaility(
                    data=self.local_outer_data_samples,
                    model_prediction=self.local_outer_model_prediction
                )
            else:
                self.write_log_file("Loaded outer Likelihood prob")
                self.write_log_file("----------------")

            self.save_restart_data(
                data=self.local_outer_prior_samples,
                data_type="outer",
                label="outer_prior_samples"
            )
            self.save_restart_data(
                data=self.local_outer_data_samples,
                data_type="outer",
                label="outer_data_samples"
            )

            self.save_restart_data(
                data=self.local_outer_likelihood_prob,
                data_type="outer",
                label="outer_likelihood_prob"
            )

            self.outer_data_computation_req_flag = False

        local_log_likelihood = np.log(self.local_outer_likelihood_prob)

        # Estimate p(y|theta_{-i})
        individual_parameter_counter, is_individual_parameter_counter_avail = self.load_restart_data(
            data_type="inner",
            label="individual_parameter_counter",
            sub_type="ICMI"
        )
        if is_individual_parameter_counter_avail is False:
            individual_parameter_counter = 0
            lower_parameter_loop_idx = 0
        else:
            lower_parameter_loop_idx = individual_parameter_counter.item()

        for iparameter in range(lower_parameter_loop_idx, self.num_parameters):
            self.write_log_file(
                "Computing I(theta_{};Y| theta_[rest of parameters])".format(iparameter))

            parameter_pair = np.array([iparameter])
            local_individual_likelihood = self.estimate_individual_likelihood(
                parameter_pair=parameter_pair,
                use_quadrature=use_quadrature,
                num_gaussian_quad_pts=single_integral_gaussian_quad_pts,
                case_type="individual"
            )

            local_log_individual_likelihood = np.log(
                local_individual_likelihood)
            self.write_log_file("Done computing log individual likelihood")

            comm.Barrier()
            global_log_likelihood = comm.gather(local_log_likelihood, root=0)
            global_log_individual_likelihood = comm.gather(
                local_log_individual_likelihood, root=0)

            if rank == 0:
                summation = sum([np.sum(global_log_likelihood[ii] -
                                global_log_individual_likelihood[ii]) for ii in range(size)])
                inidividual_mutual_information[iparameter] = (
                    1/self.global_num_outer_samples)*summation
            self.write_log_file("----------------")

            individual_parameter_counter += 1

            self.save_restart_data(
                data=individual_parameter_counter,
                data_type="inner",
                label="individual_parameter_counter",
                sub_type="ICMI"
            )

            self.save_quantity(
                "estimated_individual_mutual_information.npy", inidividual_mutual_information)

        self.write_log_file(
            ">>> End computing the individual parameter mutual information, I(theta_i;Y)")

    def compute_posterior_pair_parameter_mutual_information(self, use_quadrature=False, single_integral_gaussian_quad_pts=60, double_integral_gaussian_quad_pts=30):
        """Function computes the posterior mutual information between parameters, I(theta_i;theta_j|Y, theta_k)"""
        # Definitions
        parameter_combinations = np.array(
            list(combinations(np.arange(self.num_parameters), 2)))

        if os.path.exists("estimated_pair_mutual_information.npy"):
            pair_mutual_information = np.load(
                "estimated_pair_mutual_information.npy")
        else:
            pair_mutual_information = np.nan * \
                np.ones(parameter_combinations.shape[0])

        self.write_log_file(
            ">>>-----------------------------------------------------------<<<")
        self.write_log_file(
            ">>> Begin computing the pair parameter mutual information, I(theta_i;theta_j|Y, theta_[rest of parameters])")
        self.write_log_file(
            ">>>-----------------------------------------------------------<<<\n")

        if self.outer_data_computation_req_flag is True:
            # Generate the prior samples ~ p(\theta)
            self.local_outer_prior_samples, is_local_outer_prior_samp_avail = self.load_restart_data(
                data_type="outer",
                label="outer_prior_samples"
            )
            if is_local_outer_prior_samp_avail is False:
                self.local_outer_prior_samples = self.sample_prior_distribution(
                    num_samples=self.local_num_outer_samples)
            else:
                self.write_log_file("Done Loading outer Prior samples")
                self.write_log_file("----------------")

            # Generate data samples from the conditional distribution, p(y | \theta)
            self.local_outer_data_samples, is_local_outer_data_samp_avail = self.load_restart_data(
                data_type="outer",
                label="outer_data_samples"
            )
            if is_local_outer_data_samp_avail is False:
                self.local_outer_data_samples, self.local_outer_model_prediction = self.sample_likelihood(
                    theta=self.local_outer_prior_samples)
                self.write_log_file("Done computing outer data samples")
                self.write_log_file("----------------")
            else:
                self.write_log_file("Done Loading outer data samples")
                self.write_log_file("----------------")

            # Likelihood probability
            self.local_outer_likelihood_prob, is_local_outer_likelihood_prob_avail = self.load_restart_data(
                data_type="outer",
                label="outer_likelihood_prob"
            )

            if is_local_outer_likelihood_prob_avail is False:
                self.local_outer_likelihood_prob = self.evaluate_likelihood_probaility(
                    data=self.local_outer_data_samples,
                    model_prediction=self.local_outer_model_prediction
                )
            else:
                self.write_log_file("Done Loading outer Likelihood samples")
                self.write_log_file("----------------")

            self.save_restart_data(
                data=self.local_outer_prior_samples,
                data_type="outer",
                label="outer_prior_samples"
            )
            self.save_restart_data(
                data=self.local_outer_data_samples,
                data_type="outer",
                label="outer_data_samples"
            )

            self.save_restart_data(
                data=self.local_outer_likelihood_prob,
                data_type="outer",
                label="outer_likelihood_prob"
            )

            self.outer_data_computation_req_flag = False

        local_log_likelihood = np.log(self.local_outer_likelihood_prob)

        pair_parameter_counter, is_pair_parameter_counter_avail = self.load_restart_data(
            data_type="inner",
            label="pair_parameter_counter",
            sub_type="PCMI"
        )

        if is_pair_parameter_counter_avail is False:
            pair_parameter_counter = 0
            lower_parameter_pair_loop_idx = 0
        else:
            lower_parameter_pair_loop_idx = pair_parameter_counter.item()

        for jj, parameter_pair in enumerate(parameter_combinations[lower_parameter_pair_loop_idx:, :]):

            self.write_log_file(
                ">>> Begin Pair {}/{} computations \n".format(np.arange(parameter_combinations.shape[0])[lower_parameter_pair_loop_idx:][jj]+1, parameter_combinations.shape[0]))

            individual_likelihood_parameter_counter, is_individual_likelihood_parameter_counter_avail = self.load_restart_data(
                data_type="inner",
                label="pair_outer_counter_individual_likelihood_parameter_pair_{}_{}".format(
                    parameter_pair[0], parameter_pair[1]),
                sub_type="PCMI"
            )

            local_individual_likelihood, is_local_individual_likelihood_avail = self.load_restart_data(
                data_type="inner",
                label="pair_outer_individual_likelihood_parameter_pair_{}_{}".format(
                    parameter_pair[0], parameter_pair[1]),
                sub_type="PCMI"
            )

            if is_individual_likelihood_parameter_counter_avail is False:
                individual_likelihood_parameter_counter = 0
                lower_parameter_loop_idx = 0
            else:
                lower_parameter_loop_idx = individual_likelihood_parameter_counter.item()

            if is_local_individual_likelihood_avail is False:
                local_individual_likelihood = np.nan * \
                    np.ones((2, self.local_num_outer_samples))

            for ii, iparameter in enumerate(parameter_pair[lower_parameter_loop_idx:]):

                self.write_log_file("Computing individual likelihood (sampling theta_{}), p(y|theta_{}, theta_[rest of parameters])".format(
                    iparameter, parameter_pair[parameter_pair != iparameter]))

                index = np.arange(2)[lower_parameter_loop_idx:][ii]

                local_individual_likelihood[index, :] = self.estimate_individual_likelihood(
                    parameter_pair=np.array([iparameter]),
                    use_quadrature=use_quadrature,
                    num_gaussian_quad_pts=single_integral_gaussian_quad_pts,
                    case_type="pair",
                    parent_pair=parameter_pair
                )

                individual_likelihood_parameter_counter += 1

                self.save_restart_data(
                    data=local_individual_likelihood,
                    data_type="inner",
                    label="pair_outer_individual_likelihood_parameter_pair_{}_{}".format(
                        parameter_pair[0], parameter_pair[1]),
                    sub_type="PCMI"
                )

                self.save_restart_data(
                    data=individual_likelihood_parameter_counter,
                    data_type="inner",
                    label="pair_outer_counter_individual_likelihood_parameter_pair_{}_{}".format(
                        parameter_pair[0], parameter_pair[1]),
                    sub_type="PCMI"
                )

                self.write_log_file("End computing individual likelihood (sampling theta_{}), p(y|theta_{}, theta_[rest of parameters])".format(
                    iparameter, parameter_pair[parameter_pair != iparameter]))
                self.write_log_file("----------------")

            local_log_individual_likelihood = np.log(
                local_individual_likelihood)

            self.write_log_file("Computing pair likelihood (sampling theta_{} and theta_{}), p(y|theta_[rest of parameters])".format(
                parameter_pair[0], parameter_pair[1]))

            local_pair_likelihood = self.estimate_individual_likelihood(
                parameter_pair=parameter_pair,
                use_quadrature=use_quadrature,
                num_gaussian_quad_pts=double_integral_gaussian_quad_pts,
                case_type="pair",
                parent_pair=parameter_pair
            )

            self.write_log_file("----------------")
            local_log_pair_likelihood = np.log(local_pair_likelihood)
            self.write_log_file("End computing pair likelihood (sampling theta_{} and theta_{}), p(y|theta_[rest of parameters])\n".format(
                parameter_pair[0], parameter_pair[1]))

            comm.Barrier()

            global_log_likelihood = comm.gather(local_log_likelihood, root=0)
            global_log_individual_likelihood = comm.gather(
                local_log_individual_likelihood, root=0)
            global_log_pair_likelihood = comm.gather(
                local_log_pair_likelihood, root=0)

            if rank == 0:
                summation = sum([np.sum(global_log_likelihood[ii]
                                        + global_log_pair_likelihood[ii]
                                        - np.sum(global_log_individual_likelihood[ii], axis=0))
                                 for ii in range(size)])
                pair_mutual_information[np.arange(parameter_combinations.shape[0])[lower_parameter_pair_loop_idx:][jj]] = (
                    1/self.global_num_outer_samples)*summation

            self.write_log_file(
                ">>> End Pair {}/{} computations\n".format(np.arange(parameter_combinations.shape[0])[lower_parameter_pair_loop_idx:][jj]+1, parameter_combinations.shape[0]))

            self.save_quantity(
                "estimated_pair_mutual_information.npy", pair_mutual_information)

            pair_parameter_counter += 1

            self.save_restart_data(
                data=pair_parameter_counter,
                data_type="inner",
                label="pair_parameter_counter",
                sub_type="PCMI"
            )

    def estimate_pair_likelihood(self, parameter_pair, use_quadrature, quadrature_rule="gaussian"):
        """Function commputes the pair likelihood defined as p(y|theta_{k})"""
        if use_quadrature is True:
            individual_likelihood_prob = self.integrate_individual_likelihood_via_quadrature(
                quadrature_rule=quadrature_rule,
                parameter_pair=parameter_pair
            )
        else:
            individual_likelihood_prob = self.integrate_individual_likelihood_via_mc(
                parameter_pair=parameter_pair
            )
        return individual_likelihood_prob

    def estimate_individual_likelihood(self, num_gaussian_quad_pts, parameter_pair, use_quadrature, case_type, quadrature_rule="gaussian", parent_pair=None):
        """Function commputes the individual likelihood defined as p(y|theta_{-i})"""
        if use_quadrature is True:
            individual_likelihood_prob = self.integrate_individual_likelihood_via_quadrature(
                quadrature_rule=quadrature_rule,
                parameter_pair=parameter_pair,
                num_gaussian_quad_pts=num_gaussian_quad_pts,
                case_type=case_type,
                parent_pair=parent_pair,
                importance_sampling_cov_factors=[1e-2, 1e-6, 1e-9, 1e-10],
            )
        else:
            assert(self.restart is False), "Error"
            individual_likelihood_prob = self.integrate_individual_likelihood_via_mc(
                parameter_pair=parameter_pair
            )
        return individual_likelihood_prob

    def integrate_individual_likelihood_via_mc(self, parameter_pair):
        """Function integrates the individual likelihood via Monte-Carlo"""
        # Definitions
        individual_likelihood_prob = np.zeros(self.local_num_outer_samples)
        fixed_parameter_idx = self.get_fixed_parameter_id(
            parameter_pair=parameter_pair)
        sample_parameter_idx = self.get_sample_parameter_id(
            parameter_pair=parameter_pair)
        pre_exp = 1 / ((2*np.pi*self.model_noise_cov_scalar)
                       ** (self.spatial_res/2))

        # Extract sample parameter samples
        inner_parameter_samples = self.sample_parameter_distribution(
            num_samples=self.local_num_inner_samples,
            parameter_pair=sample_parameter_idx
        )

        for isample in range(self.local_num_outer_samples):
            outer_sample = np.expand_dims(
                self.local_outer_data_samples[:, :, isample], axis=-1)

            fixed_parameters = self.local_outer_prior_samples[fixed_parameter_idx, isample].reshape(
                -1, 1)
            eval_parameters = np.zeros(
                (self.num_parameters, self.local_num_inner_samples))
            eval_parameters[fixed_parameter_idx, :] = np.tile(
                fixed_parameters, (1, self.local_num_inner_samples))
            eval_parameters[sample_parameter_idx, :] = inner_parameter_samples

            inner_model_prediction = self.compute_model_prediction(
                theta=eval_parameters)

            error = outer_sample - inner_model_prediction
            error_norm_sq = np.linalg.norm(error, axis=1)**2

            sample_individual_likelihood = pre_exp * \
                np.exp(-0.5*np.sum(error_norm_sq /
                                   self.model_noise_cov_scalar, axis=0))

            individual_likelihood_prob[isample] = (
                1/self.local_num_inner_samples)*np.sum(sample_individual_likelihood)

        return individual_likelihood_prob

    def integrate_individual_likelihood_via_quadrature(self, parameter_pair, num_gaussian_quad_pts, case_type, quadrature_rule="gaussian", parent_pair=None, importance_sampling_cov_factors=None):
        """Function integrates the individual likelihood via Quadrature"""
        # Definitions
        fixed_parameter_idx = self.get_fixed_parameter_id(
            parameter_pair=parameter_pair)
        sample_parameter_idx = self.get_sample_parameter_id(
            parameter_pair=parameter_pair)

        if case_type == "individual":
            likelihood_file_name = "individual_inner_likelihood_parameter_" + \
                str(sample_parameter_idx.item())
            counter_file_name = "individual_inner_counter_parameter_" + \
                str(sample_parameter_idx.item())
            sub_type = "ICMI"
        elif case_type == "pair":
            if parameter_pair.shape[0] == 1:
                likelihood_file_name = "pair_inner_likelihood_parameter_pair_{}_{}_sample_parameter_{}".format(
                    parent_pair[0], parent_pair[1], sample_parameter_idx.item())
                counter_file_name = "pair_inner_counter_parameter_pair_{}_{}_sample_parameter_{}".format(
                    parent_pair[0], parent_pair[1], sample_parameter_idx.item())
                sub_type = "PCMI"
            elif parameter_pair.shape[0] == 2:
                likelihood_file_name = "pair_inner_likelihood_parameter_pair_{}_{}_sample_parameters_{}_and_{}".format(
                    parent_pair[0], parent_pair[1], parent_pair[0], parent_pair[1])
                counter_file_name = "pair_inner_counter_parameter_pair_{}_{}_sample_parameters_{}_and_{}".format(
                    parent_pair[0], parent_pair[1], parent_pair[0], parent_pair[1])
                sub_type = "PCMI"
            else:
                raise ValueError("Invalid selection")
        else:
            raise ValueError("Invalid selection")

        individual_likelihood_prob, is_individual_likelihood_prob_avail = self.load_restart_data(
            data_type="inner",
            label=likelihood_file_name,
            sub_type=sub_type
        )

        local_conditional_mutual_information_counter, is_local_conditional_mutual_information_counter_avail = self.load_restart_data(
            data_type="inner",
            label=counter_file_name,
            sub_type=sub_type
        )

        if is_individual_likelihood_prob_avail is False:
            individual_likelihood_prob = np.nan * \
                np.ones(self.local_num_outer_samples)

        if is_local_conditional_mutual_information_counter_avail is False:
            local_conditional_mutual_information_counter = 0
            local_lower_loop_idx = 0
        else:
            local_lower_loop_idx = local_conditional_mutual_information_counter.item()

        sample_parameter_mean, sample_parameter_cov = self.get_selected_parameter_stats(
            parameter_pair=sample_parameter_idx)

        def target_density_estimator(eval_points):
            probability = self.compute_gaussian_prob(
                mean=sample_parameter_mean,
                cov=sample_parameter_cov,
                samples=eval_points
            )

            return probability
        
        for isample in range(local_lower_loop_idx, self.local_num_outer_samples):
            # Extract outer sample
            outer_sample = np.expand_dims(
                self.local_outer_data_samples[:, :, isample], axis=-1)
            # Fixed parameters
            fixed_parameters = self.local_outer_prior_samples[fixed_parameter_idx, isample].reshape(
                -1, 1)

            # Integrand
            def integrand(eval_theta):
                integrand_val = self.pair_wise_likelihood_integrand(
                    theta=eval_theta,
                    fixed_parameters=fixed_parameters,
                    outer_sample=outer_sample,
                    parameter_pair=parameter_pair
                )
                return integrand_val

            if quadrature_rule == "unscented":

                # Use importance sampling
                for _, ifactor in enumerate(importance_sampling_cov_factors):

                    cov_multiplication_factor = ifactor*np.eye(sample_parameter_mean.shape[0])


                    unscented_quad = unscented_quadrature(
                        mean=self.get_sample_centered_mean(sample_parameter_extracted_sample=self.local_outer_prior_samples[sample_parameter_idx, isample]),
                        cov=sample_parameter_cov*cov_multiplication_factor,
                        integrand=integrand
                    )

                    def proposal_density_estimator(eval_points):
                                                    probability = self.compute_gaussian_prob(
                                                        mean=self.get_sample_centered_mean(sample_parameter_extracted_sample=self.local_outer_prior_samples[sample_parameter_idx, isample]),
                                                        cov=sample_parameter_cov*cov_multiplication_factor,
                                                        samples=eval_points
                                                    )
                                                    return probability


                    quadrature_points = unscented_quad.compute_quadrature_points()

                    importance_sampling_weights = self.compute_importance_sampling_weights(
                        target_density_estimator=target_density_estimator,
                        proposal_density_estimator=proposal_density_estimator,
                        eval_points=quadrature_points
                    )

                    individual_likelihood_mean, individual_likelihood_cov = unscented_quad.compute_integeral(
                        importance_sampling_weights=importance_sampling_weights
                    )

                    individual_likelihood_prob[isample] = individual_likelihood_mean + \
                        np.sqrt(individual_likelihood_cov)*np.random.randn(1)

                    if individual_likelihood_prob[isample] != 0: 
                                                break

            elif quadrature_rule == "gaussian":

                for _, ifactor in enumerate(importance_sampling_cov_factors):

                    cov_multiplication_factor = ifactor*np.eye(sample_parameter_mean.shape[0])

                    gh = gauss_hermite_quadrature(
                        mean=self.get_sample_centered_mean(sample_parameter_extracted_sample=self.local_outer_prior_samples[sample_parameter_idx, isample]),
                        cov=sample_parameter_cov*cov_multiplication_factor,
                        integrand=integrand,
                        num_points=num_gaussian_quad_pts
                    )

                    def proposal_density_estimator(eval_points):
                                probability = self.compute_gaussian_prob(
                                    mean=self.get_sample_centered_mean(sample_parameter_extracted_sample=self.local_outer_prior_samples[sample_parameter_idx, isample]),
                                    cov=sample_parameter_cov*cov_multiplication_factor,
                                    samples=eval_points
                                )
                                return probability

                    quadrature_points, _ = gh.compute_quadrature_points_and_weights()

                    importance_sampling_weights = self.compute_importance_sampling_weights(
                        target_density_estimator=target_density_estimator,
                        proposal_density_estimator=proposal_density_estimator,
                        eval_points=quadrature_points
                    )

                    individual_likelihood_prob[isample] = gh.compute_integeral(
                        importance_sampling_weights=importance_sampling_weights
                    )
                    if individual_likelihood_prob[isample] != 0: 
                        break

            else:
                raise ValueError("Invalid quadrature rule")

            local_conditional_mutual_information_counter += 1

            save_condition = isample % self.conditional_mutual_information_save_restart_freq

            if (save_condition == 0) or (isample == self.local_num_outer_samples-1):

                self.save_restart_data(
                    data=individual_likelihood_prob,
                    data_type="inner",
                    label=likelihood_file_name,
                    sub_type=sub_type
                )

                self.save_restart_data(
                    data=local_conditional_mutual_information_counter,
                    data_type="inner",
                    label=counter_file_name,
                    sub_type=sub_type
                )

        return individual_likelihood_prob

    def pair_wise_likelihood_integrand(self, theta, fixed_parameters, outer_sample, parameter_pair):
        """Function returns the conditional likelihood integrand"""
        # Definitions
        num_eval_parameters = theta.shape[1]
        pre_exp = 1 / ((2*np.pi*self.model_noise_cov_scalar)
                       ** (self.spatial_res/2))
        fixed_parameter_idx = self.get_fixed_parameter_id(
            parameter_pair=parameter_pair)
        sample_parameter_idx = self.get_sample_parameter_id(
            parameter_pair=parameter_pair)

        eval_parameters = np.zeros((self.num_parameters, num_eval_parameters))

        eval_parameters[sample_parameter_idx, :] = theta
        eval_parameters[fixed_parameter_idx, :] = np.tile(
            fixed_parameters, (1, num_eval_parameters))

        model_prediction = self.compute_model_prediction(theta=eval_parameters)

        error = outer_sample - model_prediction
        error_norm_sq = np.linalg.norm(error, axis=1)**2
        sample_evidence_estimates = (pre_exp**self.num_data_samples) * \
            np.exp(-0.5*np.sum(error_norm_sq /
                               self.model_noise_cov_scalar, axis=0))

        return sample_evidence_estimates.reshape(1, -1)

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

    def get_sample_centered_mean(self, sample_parameter_extracted_sample):
        """Function constructs the mean around the sample parameter"""
        return sample_parameter_extracted_sample.reshape(-1, 1)

    def compute_gaussian_prob(self, mean, cov, samples):
        """Function evaluates the proabability of the samples given the normal distribution"""
        d = mean.shape[0]
        error = mean-samples
        exp_term = np.exp(-0.5*np.diag(error.T@np.linalg.solve(cov, error)))
        pre_exp_term = 1/(((2*np.pi)**(d/2))*(np.linalg.det(cov)**(0.5)))
        return pre_exp_term*exp_term

    def compute_importance_sampling_weights(self, target_density_estimator, proposal_density_estimator, eval_points):
        """Function computes the weights for importace sampling"""
        target_denstiy = target_density_estimator(eval_points)
        proposal_density = proposal_density_estimator(eval_points)
        return target_denstiy/proposal_density
