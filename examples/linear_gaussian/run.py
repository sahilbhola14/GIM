# from sample_based_mutual_information import approx_mutual_information
import numpy as np
from mpi4py import MPI
import sys
import os
import matplotlib.pyplot as plt
from mpi4py import MPI
from scipy.optimize import minimize
from itertools import combinations
sys.path.append("/home/sbhola/Documents/CASLAB/GIM/information_metrics")
from compute_identifiability import mutual_information, conditional_mutual_information

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class linear_gaussian():
    def __init__(self, xtrain, model_noise_cov_scalar, true_theta, prior_mean, prior_cov, global_num_outer_samples, global_num_inner_samples):
        self.xtrain = xtrain
        self.spatial_res = self.xtrain.shape[0]
        self.model_noise_cov_scalar = model_noise_cov_scalar
        self.model_noise_cov_mat = self.model_noise_cov_scalar * \
            np.eye(self.spatial_res)
        self.true_theta = true_theta
        self.num_parameters = self.true_theta.shape[0]
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.global_num_outer_samples = global_num_outer_samples
        self.global_num_inner_samples = global_num_inner_samples

        self.vm = self.compute_vander_monde_matrix()
        self.ytrain = self.compute_model_prediction(theta=self.true_theta)
        noise = np.sqrt(model_noise_cov_scalar) * \
            np.random.randn(
                self.spatial_res).reshape(1, self.spatial_res)
        self.ytrain += noise

        if rank == 0:
            self.log_file = open("log_file.dat", "w")
        else:
            self.log_file = None

    def compute_vander_monde_matrix(self):
        """Function computes the vander mode matrix"""
        vm = np.tile(self.xtrain[:, None], (1, self.num_parameters))
        vm = np.cumprod(vm, axis=1)
        return vm

    def compute_model_prediction(self, theta):
        """Function computes the model prediction"""
        prediction = self.vm@theta[:, None]
        return prediction.reshape(1, -1)

    def compute_log_likelihood(self, theta):
        """Function computes the log likelihood"""
        prediction = self.compute_model_prediction(theta)
        error = self.ytrain - prediction
        error_norm_sq = np.linalg.norm(error)**2
        return -0.5*error_norm_sq/self.model_noise_cov_scalar

    def compute_mle(self):
        """Function computes the mle estimate"""
        def objective_function(theta):
            return -self.compute_log_likelihood(theta)

        res = minimize(objective_function, np.random.randn(
            self.num_parameters), method="Nelder-Mead")
        res = minimize(objective_function, res.x)

        return res.x

    def compute_gaussian_entropy(self, cov):
        """Function computes the entropy of a gaussian distribution"""
        d = cov.shape[0]
        return 0.5*d*np.log(2*np.pi*np.exp(1)) + 0.5*np.log(np.linalg.det(cov))

    def samples_gaussian(self, mean, cov, num_samples):
        """Funciton samples form a gaussian distribution"""
        d = cov.shape[0]
        cholesky = np.linalg.cholesky(cov+np.eye(d)*1e-8)
        noise = cholesky@np.random.randn(d*num_samples).reshape(d, num_samples)
        return mean + noise

    def compute_evidence_stats(self, itheta=None):
        """Fuction computes the evidence stats"""
        if itheta is None:
            evidence_mean = self.compute_model_prediction(
                theta=self.prior_mean.ravel())
            evidence_cov = self.vm@self.prior_cov@self.vm.T + self.model_noise_cov_mat
        else:
            evidence_mean = self.compute_model_prediction(
                theta=self.prior_mean.ravel())

            vm_selected = self.vm[:, itheta].reshape(-1, 1)
            select_prior_cov = (np.diag(self.prior_cov)[itheta]).reshape(1, 1)
            evidence_cov = vm_selected@select_prior_cov@vm_selected.T

        return evidence_mean, evidence_cov

    def compute_true_mutual_information(self):
        """Function comptues the true mutual information"""
        evidence_mean, evidence_cov = self.compute_evidence_stats()
        evidence_entropy = self.compute_gaussian_entropy(cov=evidence_cov)
        print("evidence entropy : {}".format(evidence_entropy))

        prior_entropy = self.compute_gaussian_entropy(cov=self.prior_cov)
        print("prior entropy : {}".format(prior_entropy))

        correlation = self.prior_cov@self.vm.T
        d = self.num_parameters + self.spatial_res
        joint_mean = np.zeros((d, 1))
        joint_cov = np.zeros((d, d))

        joint_mean[:self.num_parameters, :] = self.prior_mean
        joint_mean[self.num_parameters:, :] = evidence_mean.T

        joint_cov[:self.num_parameters, :self.num_parameters] = self.prior_cov
        joint_cov[:self.num_parameters, self.num_parameters:] = correlation
        joint_cov[self.num_parameters:, :self.num_parameters] = correlation.T
        joint_cov[self.num_parameters:, self.num_parameters:] = evidence_cov

        joint_entropy = self.compute_gaussian_entropy(cov=joint_cov)
        print("joint entropy : {}".format(joint_entropy))

        mutual_information = prior_entropy - joint_entropy + evidence_entropy
        print("Theoretical mutual informaiton : {}".format(mutual_information))

    def compute_true_individual_parameter_data_mutual_information(self):
        """Function computes the true conditional mutual information for each parameter"""
        # Definitions
        individual_mutual_information = np.zeros(self.num_parameters)
        evidence_mean, evidence_cov = self.compute_evidence_stats()
        total_corr = self.compute_correlation(
            parameter_pair=np.arange(self.num_parameters))
        total_joint_mean, total_joint_cov = self.build_joint(
            parameter_mean=self.prior_mean,
            parameter_cov=self.prior_cov,
            evidence_mean=evidence_mean,
            evidence_cov=evidence_cov,
            correlation=total_corr
        )
        total_joint_entropy = self.compute_gaussian_entropy(
            cov=total_joint_cov)

        for iparameter in range(self.num_parameters):
            parameter_pair = np.array([iparameter])
            fixed_parameter_id = np.arange(self.num_parameters) != iparameter

            selected_parameter_mean = self.prior_mean[parameter_pair].reshape(
                parameter_pair.shape[0], 1)
            selected_parameter_cov = np.diag(
                np.diag(self.prior_cov)[parameter_pair])

            fixed_parameter_mean = self.prior_mean[fixed_parameter_id].reshape(
                sum(fixed_parameter_id), 1)
            fixed_parameter_cov = np.diag(
                np.diag(self.prior_cov)[fixed_parameter_id])

            prior_entropy = self.compute_gaussian_entropy(
                cov=selected_parameter_cov)

            correlation = self.compute_correlation(
                parameter_pair=fixed_parameter_id)

            individual_joint_mean, individual_joint_cov = self.build_joint(
                parameter_mean=fixed_parameter_mean,
                parameter_cov=fixed_parameter_cov,
                evidence_mean=evidence_mean,
                evidence_cov=evidence_cov,
                correlation=correlation
            )
            individual_joint_entropy = self.compute_gaussian_entropy(
                cov=individual_joint_cov)
            individual_mutual_information[iparameter] = prior_entropy - \
                total_joint_entropy + individual_joint_entropy

        print("True individual mutual information I(theta_i;Y|theta_not_i) : {}".format(
            individual_mutual_information))

    def compute_correlation(self, parameter_pair):
        """Function computes the correlation"""
        cov = np.diag(np.diag(self.prior_cov)[parameter_pair])
        vm_selected = self.vm[:, parameter_pair]
        return cov@vm_selected.T

    def build_joint(self, parameter_mean, parameter_cov, evidence_mean, evidence_cov, correlation):
        """Function builds the joint"""
        num_parameters = parameter_mean.shape[0]
        dim = num_parameters + self.spatial_res
        joint_mean = np.zeros((dim, 1))
        joint_cov = np.zeros((dim, dim))
        joint_mean[:num_parameters, :] = parameter_mean
        joint_mean[num_parameters:, :] = evidence_mean.T
        joint_cov[:num_parameters, :num_parameters] = parameter_cov
        joint_cov[:num_parameters, num_parameters:] = correlation
        joint_cov[num_parameters:, :num_parameters] = correlation.T
        joint_cov[num_parameters:, num_parameters:] = evidence_cov

        return joint_mean, joint_cov

    def display_messsage(self, message, print_rank="root"):
        """Print function"""
        if print_rank == "root":
            if rank == 0:
                print(message)
        elif print_rank == "all":
            print(message+" at rank : {}".format(rank))
        elif rank == print_rank:
            print(message+" at rank : {}".format(rank))

    def estimate_mutual_information(self):
        """Function estimates the mutual information"""
        estimator = mutual_information(
                forward_model=self.compute_model_prediction,
                prior_mean=self.prior_mean,
                prior_cov=self.prior_cov,
                model_noise_cov_scalar=self.model_noise_cov_scalar,
                global_num_outer_samples=self.global_num_outer_samples,
                global_num_inner_samples=self.global_num_inner_samples,
                save_path=os.getcwd(),
                restart=False,
                log_file=self.log_file,
                ytrain=self.ytrain
                )

        estimator.estimate_mutual_information_via_mc(
                use_quadrature=True
                )

    def estimate_conditional_mutual_information(self):
        """Function estimates the conditional_mutual_information"""

        estimator = conditional_mutual_information(
                forward_model=self.compute_model_prediction,
                prior_mean=self.prior_mean,
                prior_cov=self.prior_cov,
                model_noise_cov_scalar=self.model_noise_cov_scalar,
                global_num_outer_samples=self.global_num_outer_samples,
                global_num_inner_samples=self.global_num_inner_samples,
                save_path=os.getcwd(),
                restart=False,
                log_file=self.log_file,
                ytrain=self.ytrain
                )

        estimator.compute_individual_parameter_data_mutual_information_via_mc(
                use_quadrature=True,
                single_integral_gaussian_quad_pts=50
                )

        estimator.compute_posterior_pair_parameter_mutual_information(
                use_quadrature=True,
                single_integral_gaussian_quad_pts=50,
                double_integral_gaussian_quad_pts=50
                )

    def compute_true_evidence_entropy(self):
        evidence_mean, evidence_cov = self.compute_evidence_stats()
        theta = np.load("eval_theta.npy")
        prediction = []
        for ii in range(theta.shape[1]):
            prediction.append(
                self.compute_model_prediction(theta=theta[:, ii]))
        prediction = np.array(prediction).T
        noise = np.sqrt(self.model_noise_cov_scalar) * \
            np.random.randn(np.prod(prediction.shape)
                            ).reshape(prediction.shape)
        data = prediction + noise
        error = np.expand_dims(evidence_mean.T, axis=-1) - data
        error_norm_sq = np.linalg.norm(error, axis=1)**2
        exp_term = np.exp(-0.5*np.sum(error_norm_sq /
                          self.model_noise_cov_scalar, axis=0))
        pre_exp = 1 / ((2*np.pi*self.model_noise_cov_scalar)
                       ** (error.shape[1]/2))
        evidence = pre_exp*exp_term
        np.save("true_evidence.npy", evidence)

    def compute_true_pair_parameter_data_mutual_information(self):
        """Function computes the true entropy between the parameters given the data"""
        evidence_mean, evidence_cov = self.compute_evidence_stats()
        parameter_combinations = np.array(
            list(combinations(np.arange(self.num_parameters), 2)))

        def get_fixed_parameter_id(x):
            parameter_id = np.arange(self.num_parameters)
            return ~np.logical_or(parameter_id == x[0], parameter_id == x[1])

        def get_selected_parameter_joint_id(x, select_id):
            parameter_id = np.arange(self.num_parameters)
            condition_1 = ~np.logical_or(parameter_id == x[0], parameter_id == x[1]) 
            condition_2 = parameter_id == select_id
            return np.logical_or(condition_1, condition_2)

        for iparameter in parameter_combinations:
            # h(y, \theta_k)
            fixed_parameter_mean = self.prior_mean[get_fixed_parameter_id(iparameter), :]
            fixed_parameter_cov = np.diag(np.diag(self.prior_cov)[
                                          get_fixed_parameter_id(iparameter)])
            correlation_fixed_param_data = self.compute_correlation(parameter_pair=get_fixed_parameter_id(iparameter))
            joint_fixed_param_mean, joint_fixed_param_cov = self.build_joint(
                    parameter_mean=fixed_parameter_mean,
                    parameter_cov=fixed_parameter_cov,
                    evidence_mean=evidence_mean,
                    evidence_cov=evidence_cov,
                    correlation=correlation_fixed_param_data
                    )
            joint_fixed_param_entropy = self.compute_gaussian_entropy(cov=joint_fixed_param_cov)
            print("h(Y, theta_k) : {}".format(joint_fixed_param_entropy))

            # h(y, theta_i, theta_k)
            selected_parameter_id = get_selected_parameter_joint_id(iparameter, iparameter[0])
            selected_parameter_mean = self.prior_mean[selected_parameter_id, :]
            selected_parameter_cov = np.diag(np.diag(self.prior_cov)[selected_parameter_id])
            correlation_pair_param_data = self.compute_correlation(parameter_pair=selected_parameter_id)

            joint_pair_param_mean, joint_pair_param_cov = self.build_joint(
                    parameter_mean=selected_parameter_mean,
                    parameter_cov=selected_parameter_cov,
                    evidence_mean=evidence_mean,
                    evidence_cov=evidence_cov,
                    correlation=correlation_pair_param_data
                    )
            joint_pair_param_one_entropy = self.compute_gaussian_entropy(cov=joint_pair_param_cov)
            print("h(Y, theta_i, theta_k) : {}".format(joint_pair_param_one_entropy))

            # h(y, theta_j, theta_k)
            selected_parameter_id = get_selected_parameter_joint_id(iparameter, iparameter[1])
            selected_parameter_mean = self.prior_mean[selected_parameter_id, :]
            selected_parameter_cov = np.diag(np.diag(self.prior_cov)[selected_parameter_id])
            correlation_pair_param_data = self.compute_correlation(parameter_pair=selected_parameter_id)

            joint_pair_param_mean, joint_pair_param_cov = self.build_joint(
                    parameter_mean=selected_parameter_mean,
                    parameter_cov=selected_parameter_cov,
                    evidence_mean=evidence_mean,
                    evidence_cov=evidence_cov,
                    correlation=correlation_pair_param_data
                    )
            joint_pair_param_two_entropy = self.compute_gaussian_entropy(cov=joint_pair_param_cov)
            print("h(Y, theta_j, theta_k) : {}".format(joint_pair_param_two_entropy))

            # h(y, theta_i, theta_j, theta_k)
            total_correlation = self.compute_correlation(parameter_pair=np.arange(self.num_parameters))

            joint_mean, joint_cov = self.build_joint(
                    parameter_mean=self.prior_mean,
                    parameter_cov=self.prior_cov,
                    evidence_mean=evidence_mean,
                    evidence_cov=evidence_cov,
                    correlation=total_correlation
                    )
            joint_entropy = self.compute_gaussian_entropy(cov=joint_cov)
            print("h(Y, theta_i, theta_j, theta_k) : {}".format(joint_entropy))
            conditional_mutual_information = joint_pair_param_one_entropy + joint_pair_param_two_entropy - joint_fixed_param_entropy -joint_entropy
            
            print("I(theta_{};theta_{} | y, theta_[not selected]) : {}".format(iparameter[0], iparameter[1], conditional_mutual_information))
            print("------------")


def main():
    num_samples = 100
    xtrain = np.linspace(-1, 1, num_samples)
    model_noise_cov_scalar = 1e-1
    true_theta = np.arange(1, 4)
    prior_mean = true_theta.copy().reshape(-1, 1)
    prior_cov = np.eye(true_theta.shape[0])
    global_num_outer_samples = 10000
    global_num_inner_samples = 10

    model = linear_gaussian(
        xtrain=xtrain,
        model_noise_cov_scalar=model_noise_cov_scalar,
        true_theta=true_theta,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        global_num_outer_samples=global_num_outer_samples,
        global_num_inner_samples=global_num_inner_samples
    )

    # Theoretical estimates
    # model.compute_true_mutual_information()
    model.compute_true_individual_parameter_data_mutual_information()
    model.compute_true_pair_parameter_data_mutual_information()

    # Estimates
    # model.estimate_mutual_information()
    model.estimate_conditional_mutual_information()

    if rank == 0:
        model.log_file.close()


if __name__ == "__main__":
    main()
