import numpy as np
from mpi4py import MPI
import yaml
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.optimize import minimize
from itertools import combinations

sys.path.append("../../../information_metrics/")
sys.path.append("../forward_model/")
sys.path.append("../../../mcmc/")
from linear_gaussian import linear_gaussian
from compute_identifiability import conditional_mutual_information
from SobolIndex import SobolIndex
from mcmc import adaptive_metropolis_hastings
from mcmc_utils import sub_sample_data

plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=25)
plt.rc("lines", linewidth=3)
plt.rc("axes", labelpad=18, titlepad=20)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class learn_linear_gaussian:
    def __init__(self, config_data, campaign_path, prior_mean, prior_cov):
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov

        self.prior_cov = prior_cov
        self.num_parameters = self.prior_mean.shape[0]

        # Extract configurations
        self.campaign_path = campaign_path
        self.model_noise_cov = config_data["model_noise_cov"]

        training_data_set = np.load(
            os.path.join(self.campaign_path, "training_data.npy")
        )
        self.xtrain = training_data_set[:, 0]
        self.ytrain = training_data_set[:, 1].reshape(1, -1)
        self.sample_idx = np.load(os.path.join(self.campaign_path, "sample_idx.npy"))

        self.num_data_points = self.ytrain.shape[0]
        self.spatial_resolution = self.ytrain.shape[1]
        self.objective_scaling = config_data["objective_scaling"]

        # Model identifiability
        self.global_num_outer_samples = config_data["global_num_outer_samples"]
        self.global_num_inner_samples = config_data["global_num_inner_samples"]
        self.restart_identifiability = config_data["restart_identifiability"]
        if rank == 0:
            log_file_path = os.path.join(self.campaign_path, "log_file.dat")
            self.log_file = open(log_file_path, "w")
        else:
            self.log_file = None

        # Forward model
        self.linear_gaussian_model = linear_gaussian(
            spatial_res=self.spatial_resolution
        )
        self.vm = self.linear_gaussian_model.compute_vm()
        self.sub_sampled_vm = self.vm[self.sample_idx, :]

        self.sample_idx_mat = np.zeros(
            (self.spatial_resolution, config_data["total_samples"])
        )
        self.sample_idx_mat[np.arange(self.spatial_resolution), self.sample_idx] = 1
        self.forward_model = self.linear_gaussian_model.compute_prediction

    def compute_model_prediction(self, theta, proc_log_file=None):
        prediction = self.forward_model(theta=theta)[self.sample_idx, :].T
        return prediction

    def sort_prediction(self, prediction):
        sorted_idx_id = np.argsort(self.sample_idx)
        sorted_prediction = prediction[:, sorted_idx_id]
        return sorted_prediction

    def sort_input(self):
        sorted_idx_id = np.argsort(self.sample_idx)
        return self.xtrain[sorted_idx_id]

    def compute_log_likelihood(self, theta):
        """Function computes the log likelihood"""
        prediction = self.compute_model_prediction(theta)
        error = self.ytrain - prediction
        error_norm_sq = np.linalg.norm(error, axis=1) ** 2
        log_likelihood = -(0.5 / self.model_noise_cov) * np.sum(error_norm_sq, axis=0)
        return log_likelihood

    def compute_mle(self):
        """Function computes the mle estimate"""

        def negative_log_likelihood(theta):
            objective_function = -self.objective_scaling * self.compute_log_likelihood(
                theta
            )
            print("MLE objective : {0:.18e}".format(objective_function))
            return objective_function

        res = minimize(
            negative_log_likelihood,
            np.random.randn(self.num_parameters),
            method="Nelder-Mead",
        )
        res = minimize(negative_log_likelihood, res.x)
        if res.success is False:
            print("Numerical minima not found")

        theta_mle = res.x

        prediction = self.compute_model_prediction(theta=res.x)
        prediction_data = np.zeros((self.spatial_resolution, 2))
        prediction_data[:, 0] = self.xtrain
        prediction_data[:, 1] = prediction

        save_prediction_path = os.path.join(self.campaign_path, "prediction_mle.npy")
        save_mle_path = os.path.join(self.campaign_path, "theta_mle.npy")

        np.save(save_prediction_path, prediction_data)
        np.save(save_mle_path, theta_mle)

        return theta_mle

    def compute_map(self):
        """Function computes the map estimate"""
        theta_init = np.random.randn(self.num_parameters)

        def objective_function(theta):
            objective_function = (
                -self.objective_scaling * self.compute_unnormalized_posterior(theta)
            )
            print("MAP objective : {0:.18e}".format(objective_function))
            return objective_function

        res = minimize(objective_function, theta_init, method="Nelder-Mead")
        res = minimize(objective_function, res.x)

        theta_map = res.x
        theta_map_cov = res.hess_inv

        save_map_path = os.path.join(self.campaign_path, "theta_map.npy")
        save_map_cov_path = os.path.join(self.campaign_path, "theta_map_cov.npy")

        np.save(save_map_path, theta_map.reshape(-1, 1))
        np.save(save_map_cov_path, theta_map_cov)

        return theta_map.reshape(-1, 1), theta_map_cov

    def compute_unnormalized_posterior(self, theta):
        """Function computes the unnormalized log posterior"""
        unnormalized_log_likelihood = self.compute_log_likelihood(theta)
        unnormalized_log_prior = self.compute_log_prior(theta)
        unnormalized_log_posterior = (
            unnormalized_log_likelihood + unnormalized_log_prior
        )
        return unnormalized_log_posterior

    def compute_log_prior(self, theta):
        """Function computes the log prior (unnormalized)"""
        error = (theta - self.prior_mean.ravel()).reshape(-1, 1)
        exp_term_solve = error.T @ np.linalg.solve(self.prior_cov, error)
        exp_term = -0.5 * exp_term_solve
        return exp_term.item()

    def plot_mle_estimate(self, theta_mle):
        prediction = self.compute_model_prediction(theta_mle)
        prediction_true = self.compute_model_prediction(
            self.linear_gaussian_model.true_theta
        )

        sorted_prediction = self.sort_prediction(prediction)
        sorted_prediction_true = self.sort_prediction(prediction_true)

        save_fig_path = os.path.join(self.campaign_path, "Figures/prediction_mle.png")

        sorted_xtrain = self.sort_input()
        fig, axs = plt.subplots(figsize=(10, 6))
        axs.scatter(self.xtrain, self.ytrain.ravel(), label="Data", c="k", alpha=0.8)
        axs.plot(
            sorted_xtrain, sorted_prediction.ravel(), label="Prediction", color="r"
        )
        axs.plot(
            sorted_xtrain, sorted_prediction_true.ravel(), "--", label="True", color="k"
        )
        axs.grid(True, axis="both", which="major", color="k", alpha=0.5)
        axs.grid(True, axis="both", which="minor", color="grey", alpha=0.3)
        axs.set_xlabel("x")
        axs.set_ylabel("y")
        axs.legend(framealpha=1.0)
        axs.set_title("M.L.E.")
        plt.tight_layout()
        plt.savefig(save_fig_path)
        plt.close()

    def compute_evidence_stats(self, itheta=None):
        """Fuction computes the evidence stats"""
        if itheta is None:
            evidence_mean = self.compute_model_prediction(theta=self.prior_mean).T

            evidence_cov = (
                self.sub_sampled_vm @ self.prior_cov @ self.sub_sampled_vm.T
                + self.model_noise_cov * (self.sample_idx_mat @ self.sample_idx_mat.T)
            )

        else:
            evidence_mean = self.compute_model_prediction(theta=self.prior_mean.ravel())

            vm_selected = self.vm[:, itheta].reshape(-1, 1)
            select_prior_cov = (np.diag(self.prior_cov)[itheta]).reshape(1, 1)
            evidence_cov = vm_selected @ select_prior_cov @ vm_selected.T

        return evidence_mean, evidence_cov

    def compute_gaussian_entropy(self, cov):
        """Function computes the entropy of a gaussian distribution"""
        d = cov.shape[0]
        return 0.5 * d * np.log(2 * np.pi * np.exp(1)) + 0.5 * np.log(
            np.linalg.det(cov)
        )

    def compute_true_mutual_information(self):
        """Function comptues the true mutual information"""
        evidence_mean, evidence_cov = self.compute_evidence_stats()
        evidence_entropy = self.compute_gaussian_entropy(cov=evidence_cov)
        print("evidence entropy : {}".format(evidence_entropy))

        prior_entropy = self.compute_gaussian_entropy(cov=self.prior_cov)
        print("prior entropy : {}".format(prior_entropy))

        correlation = self.prior_cov @ (self.sub_sampled_vm).T
        d = self.num_parameters + self.spatial_resolution
        joint_mean = np.zeros((d, 1))
        joint_cov = np.zeros((d, d))

        joint_mean[: self.num_parameters, :] = self.prior_mean
        joint_mean[self.num_parameters :, :] = evidence_mean
        joint_cov[: self.num_parameters, : self.num_parameters] = self.prior_cov
        joint_cov[: self.num_parameters, self.num_parameters :] = correlation
        joint_cov[self.num_parameters :, : self.num_parameters] = correlation.T
        joint_cov[self.num_parameters :, self.num_parameters :] = evidence_cov
        joint_entropy = self.compute_gaussian_entropy(cov=joint_cov)
        print("joint entropy : {}".format(joint_entropy))
        mutual_information = prior_entropy - joint_entropy + evidence_entropy
        print("Theoretical mutual informaiton : {}".format(mutual_information))

    def compute_true_individual_parameter_data_mutual_information(self):
        """Function computes the true conditional mutual information for
        each parameter
        """
        # Definitions
        individual_mutual_information = np.zeros(self.num_parameters)
        evidence_mean, evidence_cov = self.compute_evidence_stats()

        total_corr = self.compute_correlation(
            parameter_pair=np.arange(self.num_parameters)
        )

        total_joint_mean, total_joint_cov = self.build_joint(
            parameter_mean=self.prior_mean,
            parameter_cov=self.prior_cov,
            evidence_mean=evidence_mean,
            evidence_cov=evidence_cov,
            correlation=total_corr,
        )

        total_joint_entropy = self.compute_gaussian_entropy(cov=total_joint_cov)

        for iparameter in range(self.num_parameters):
            parameter_pair = np.array([iparameter])
            fixed_parameter_id = np.arange(self.num_parameters) != iparameter

            # selected_parameter_mean = self.prior_mean[parameter_pair].reshape(
            #     parameter_pair.shape[0], 1
            # )
            selected_parameter_cov = np.diag(np.diag(self.prior_cov)[parameter_pair])

            fixed_parameter_mean = self.prior_mean[fixed_parameter_id].reshape(
                sum(fixed_parameter_id), 1
            )
            fixed_parameter_cov = np.diag(np.diag(self.prior_cov)[fixed_parameter_id])

            prior_entropy = self.compute_gaussian_entropy(cov=selected_parameter_cov)

            correlation = self.compute_correlation(parameter_pair=fixed_parameter_id)

            individual_joint_mean, individual_joint_cov = self.build_joint(
                parameter_mean=fixed_parameter_mean,
                parameter_cov=fixed_parameter_cov,
                evidence_mean=evidence_mean,
                evidence_cov=evidence_cov,
                correlation=correlation,
            )
            individual_joint_entropy = self.compute_gaussian_entropy(
                cov=individual_joint_cov
            )
            individual_mutual_information[iparameter] = (
                prior_entropy - total_joint_entropy + individual_joint_entropy
            )

        print(
            "True individual mutual information I(theta_i;Y|theta_not_i) : {}".format(
                individual_mutual_information
            )
        )

    def compute_true_pair_parameter_data_mutual_information(self):
        """Function computes the true entropy between the parameters given the data"""
        evidence_mean, evidence_cov = self.compute_evidence_stats()
        parameter_combinations = np.array(
            list(combinations(np.arange(self.num_parameters), 2))
        )

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
            fixed_parameter_mean = self.prior_mean[
                get_fixed_parameter_id(iparameter), :
            ]
            fixed_parameter_cov = np.diag(
                np.diag(self.prior_cov)[get_fixed_parameter_id(iparameter)]
            )
            correlation_fixed_param_data = self.compute_correlation(
                parameter_pair=get_fixed_parameter_id(iparameter)
            )
            joint_fixed_param_mean, joint_fixed_param_cov = self.build_joint(
                parameter_mean=fixed_parameter_mean,
                parameter_cov=fixed_parameter_cov,
                evidence_mean=evidence_mean,
                evidence_cov=evidence_cov,
                correlation=correlation_fixed_param_data,
            )
            joint_fixed_param_entropy = self.compute_gaussian_entropy(
                cov=joint_fixed_param_cov
            )
            print("h(Y, theta_k) : {}".format(joint_fixed_param_entropy))

            # h(y, theta_i, theta_k)
            selected_parameter_id = get_selected_parameter_joint_id(
                iparameter, iparameter[0]
            )
            selected_parameter_mean = self.prior_mean[selected_parameter_id, :]
            selected_parameter_cov = np.diag(
                np.diag(self.prior_cov)[selected_parameter_id]
            )
            correlation_pair_param_data = self.compute_correlation(
                parameter_pair=selected_parameter_id
            )

            joint_pair_param_mean, joint_pair_param_cov = self.build_joint(
                parameter_mean=selected_parameter_mean,
                parameter_cov=selected_parameter_cov,
                evidence_mean=evidence_mean,
                evidence_cov=evidence_cov,
                correlation=correlation_pair_param_data,
            )
            joint_pair_param_one_entropy = self.compute_gaussian_entropy(
                cov=joint_pair_param_cov
            )
            print("h(Y, theta_i, theta_k) : {}".format(joint_pair_param_one_entropy))

            # h(y, theta_j, theta_k)
            selected_parameter_id = get_selected_parameter_joint_id(
                iparameter, iparameter[1]
            )
            selected_parameter_mean = self.prior_mean[selected_parameter_id, :]
            selected_parameter_cov = np.diag(
                np.diag(self.prior_cov)[selected_parameter_id]
            )
            correlation_pair_param_data = self.compute_correlation(
                parameter_pair=selected_parameter_id
            )

            joint_pair_param_mean, joint_pair_param_cov = self.build_joint(
                parameter_mean=selected_parameter_mean,
                parameter_cov=selected_parameter_cov,
                evidence_mean=evidence_mean,
                evidence_cov=evidence_cov,
                correlation=correlation_pair_param_data,
            )
            joint_pair_param_two_entropy = self.compute_gaussian_entropy(
                cov=joint_pair_param_cov
            )
            print("h(Y, theta_j, theta_k) : {}".format(joint_pair_param_two_entropy))

            # h(y, theta_i, theta_j, theta_k)
            total_correlation = self.compute_correlation(
                parameter_pair=np.arange(self.num_parameters)
            )

            joint_mean, joint_cov = self.build_joint(
                parameter_mean=self.prior_mean,
                parameter_cov=self.prior_cov,
                evidence_mean=evidence_mean,
                evidence_cov=evidence_cov,
                correlation=total_correlation,
            )
            joint_entropy = self.compute_gaussian_entropy(cov=joint_cov)
            print("h(Y, theta_i, theta_j, theta_k) : {}".format(joint_entropy))
            conditional_mutual_information = (
                joint_pair_param_one_entropy
                + joint_pair_param_two_entropy
                - joint_fixed_param_entropy
                - joint_entropy
            )

            print(
                "I(theta_{};theta_{} | y, theta_[not selected]) : {}".format(
                    iparameter[0], iparameter[1], conditional_mutual_information
                )
            )
            print("------------")

    def compute_correlation(self, parameter_pair):
        """Function computes the correlation"""
        cov = np.diag(np.diag(self.prior_cov)[parameter_pair])
        vm_selected = self.sample_idx_mat @ self.vm[:, parameter_pair]
        return cov @ vm_selected.T

    def build_joint(
        self, parameter_mean, parameter_cov, evidence_mean, evidence_cov, correlation
    ):
        """Function builds the joint"""
        num_parameters = parameter_mean.shape[0]
        dim = num_parameters + evidence_mean.shape[0]
        joint_mean = np.zeros((dim, 1))
        joint_cov = np.zeros((dim, dim))
        joint_mean[:num_parameters, :] = parameter_mean
        joint_mean[num_parameters:, :] = evidence_mean
        joint_cov[:num_parameters, :num_parameters] = parameter_cov
        joint_cov[:num_parameters, num_parameters:] = correlation
        joint_cov[num_parameters:, :num_parameters] = correlation.T
        joint_cov[num_parameters:, num_parameters:] = evidence_cov

        return joint_mean, joint_cov

    def update_prior(self, theta_mean, theta_cov):
        self.prior_mean = theta_mean
        self.prior_cov = theta_cov

    def compute_esimated_mi(self):
        mi_estimator = conditional_mutual_information(
            forward_model=self.compute_model_prediction,
            prior_mean=self.prior_mean,
            prior_cov=self.prior_cov,
            model_noise_cov_scalar=self.model_noise_cov,
            global_num_outer_samples=self.global_num_outer_samples,
            global_num_inner_samples=self.global_num_inner_samples,
            save_path=self.campaign_path,
            restart=self.restart_identifiability,
            ytrain=self.ytrain,
            log_file=self.log_file,
        )

        mi_estimator.compute_individual_parameter_data_mutual_information_via_mc(
            use_quadrature=True, single_integral_gaussian_quad_pts=30
        )

        mi_estimator.compute_posterior_pair_parameter_mutual_information(
            use_quadrature=True,
            single_integral_gaussian_quad_pts=30,
            double_integral_gaussian_quad_pts=5,
        )

    def compute_sobol_indices(self):
        """Function computes the sobol indices"""

        def forward_model(theta):
            return self.compute_model_prediction(theta)

        sobol_index = SobolIndex(
            forward_model=forward_model,
            prior_mean=self.prior_mean,
            prior_cov=self.prior_cov,
            global_num_outer_samples=self.global_num_outer_samples,
            global_num_inner_samples=self.global_num_inner_samples,
            # model_noise_cov_scalar=self.model_noise_cov,
            model_noise_cov_scalar=0,
            data_shape=(self.num_data_points, self.spatial_resolution),
            write_log_file=True,
            save_path=os.path.join(self.campaign_path, "SobolIndex"),
        )

        sobol_index.comp_first_order_sobol_indices()
        sobol_index.comp_total_effect_sobol_indices()

    def plot_map_estimate(self, theta_map, theta_map_cov):
        num_samples = 1000
        theta = theta_map + np.linalg.cholesky(theta_map_cov) @ np.random.randn(
            self.num_parameters, num_samples
        )

        prediction_true = self.compute_model_prediction(
            self.linear_gaussian_model.true_theta
        )

        prediction = np.zeros(self.ytrain.shape + (num_samples,))
        for isample in range(num_samples):
            prediction[:, :, isample] = self.compute_model_prediction(
                theta=theta[:, isample]
            )

        prediction_mean = np.mean(prediction, axis=-1)
        prediction_std = np.std(prediction, axis=-1)
        sorted_prediction_mean = self.sort_prediction(prediction=prediction_mean)
        sorted_prediction_std = self.sort_prediction(prediction=prediction_std)
        sorted_prediction_true = self.sort_prediction(prediction_true)
        upper_lim = sorted_prediction_mean + sorted_prediction_std
        lower_lim = sorted_prediction_mean - sorted_prediction_std

        sorted_input = self.sort_input()

        save_fig_path = os.path.join(self.campaign_path, "Figures/prediction_map.png")
        fig, axs = plt.subplots(figsize=(12, 6))
        axs.scatter(
            self.xtrain, self.ytrain.ravel(), c="k", s=30, zorder=-1, label="Data"
        )
        axs.plot(
            sorted_input, sorted_prediction_true.ravel(), "--", label="True", color="k"
        )
        axs.plot(
            sorted_input,
            sorted_prediction_mean.ravel(),
            color="r",
            label=r"$\mu_{prediction}$",
        )
        axs.fill_between(
            sorted_input,
            upper_lim.ravel(),
            lower_lim.ravel(),
            ls="--",
            lw=2,
            alpha=0.3,
            color="r",
            label=r"$\pm\sigma$",
        )
        axs.grid(True, axis="both", which="major", color="k", alpha=0.5)
        axs.grid(True, axis="both", which="minor", color="grey", alpha=0.3)
        axs.legend(framealpha=1.0)
        axs.set_xlabel("x")
        axs.set_ylabel("y")
        axs.set_ylim([-8, 8])
        axs.yaxis.set_minor_locator(MultipleLocator(1))
        axs.xaxis.set_minor_locator(MultipleLocator(0.25))
        axs.set_title("Aggregate posterior prediction")
        plt.tight_layout()
        plt.savefig(save_fig_path)
        plt.close()

    def compute_mcmc_samples(self, theta_map, theta_map_cov):
        """Function computes the mcmc samples"""

        def compute_post(theta):
            return self.compute_unnormalized_posterior(theta)

        mcmc_sampler = adaptive_metropolis_hastings(
            initial_sample=theta_map.ravel(),
            target_log_pdf_evaluator=compute_post,
            num_samples=200000,
            adapt_sample_threshold=10000,
            initial_cov=1e-2 * theta_map_cov,
        )

        mcmc_sampler.compute_mcmc_samples(verbose=True)

        # Compute acceptance rate
        ar = mcmc_sampler.compute_acceptance_ratio()
        if rank == 0:
            print("Acceptance rate: ", ar)

        # Compute the burn in samples
        burned_samples = sub_sample_data(
            mcmc_sampler.samples, frac_burn=0.5, frac_use=0.7
        )

        np.save(
            os.path.join(
                self.campaign_path, "burned_samples_rank_{}_.npy".format(rank)
            ),
            burned_samples,
        )


def load_configuration_file(config_file_path="./config.yaml"):
    """Function loads the configuration file"""
    with open(config_file_path, "r") as config_file:
        config_data = yaml.safe_load(config_file)
    if rank == 0:
        print("Loaded %s configuration file" % (config_file_path), flush=True)
    return config_data


def main():
    prior_mean = np.zeros((3, 1))
    prior_cov = np.eye(3)

    # Load the config data
    config_data = load_configuration_file()

    # Campaign path
    campaign_path = os.path.join(
        os.getcwd(), "campaign_results/campaign_%d" % (config_data["campaign_id"])
    )

    # Leaning model

    learning_model = learn_linear_gaussian(
        config_data=config_data,
        campaign_path=campaign_path,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
    )

    if config_data["compute_mle"]:
        theta_mle = learning_model.compute_mle()
    elif config_data["compute_map"]:
        theta_map, theta_map_cov = learning_model.compute_map()
    elif config_data["compute_post"]:
        theta_map = np.load(os.path.join(campaign_path, "theta_map.npy"))
        theta_map_cov = np.load(os.path.join(campaign_path, "theta_map_cov.npy"))
        learning_model.compute_mcmc_samples(
            theta_map=theta_map, theta_map_cov=theta_map_cov
        )

    if "--plotmle" in sys.argv:
        theta_mle = np.load(os.path.join(campaign_path, "theta_mle.npy"))
        learning_model.plot_mle_estimate(
            theta_mle=theta_mle,
        )

    if "--plotmap" in sys.argv:
        theta_map = np.load(os.path.join(campaign_path, "theta_map.npy"))
        theta_map_cov = np.load(os.path.join(campaign_path, "theta_map_cov.npy"))

        learning_model.plot_map_estimate(
            theta_map=theta_map, theta_map_cov=theta_map_cov
        )

    # Information content
    if config_data["compute_identifiability"]:
        theta_map = np.load(os.path.join(campaign_path, "theta_map.npy"))
        theta_map_cov = np.load(os.path.join(campaign_path, "theta_map_cov.npy"))

        # Update the prior
        learning_model.update_prior(theta_mean=theta_map, theta_cov=theta_map_cov)

        # True MI
        # learning_model.compute_true_mutual_information()
        # learning_model.compute_true_individual_parameter_data_mutual_information()
        # learning_model.compute_true_pair_parameter_data_mutual_information()

        # Estimated MI
        # learning_model.compute_esimated_mi()

        # Sobol indices
        learning_model.compute_sobol_indices()

    if rank == 0:
        learning_model.log_file.close()


if __name__ == "__main__":
    main()
