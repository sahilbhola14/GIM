# Author: Sahil Bhola
# Date: 6/6/2022
# Description: Identifiability ananlysis of the radiative heat transfer model
# True dynamics are considered and the parameters of the emmissivity are identified
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
from mpi4py import MPI
sys.path.append("../forward_model")
sys.path.append("/home/sbhola/Documents/CASLAB/GIM/information_metrics")

from rht_true import compute_prediction
from sample_based_mutual_information import approx_mutual_information, approx_conditional_mutual_information

plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif', size=20)
matplotlib.rcParams['font.family'] = 'sans-serif'

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class inference():
    def __init__(self, xtrain, model_noise_cov_scalar, true_theta, objective_scaling, prior_mean, prior_cov, num_outer_samples, num_inner_samples):
        self.xtrain = xtrain
        self.model_noise_cov_scalar = model_noise_cov_scalar
        self.num_samples = self.xtrain.shape[0]
        self.true_theta = true_theta
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.num_parameters = self.true_theta.shape[0]
        self.objective_scaling = objective_scaling
        self.num_outer_samples = num_outer_samples
        self.num_inner_samples = num_inner_samples

        self.ytrain = self.compute_model_prediction(theta=self.true_theta)

        self.spatial_resolution = self.ytrain.shape[0]
        noise = np.sqrt(self.model_noise_cov_scalar)*np.random.randn(
            self.num_samples*(self.spatial_resolution-2)).reshape(-1, self.num_samples)
        self.ytrain[1:-1, :] += noise

        self.model_noise_cov_mat = self.model_noise_cov_scalar * \
            np.eye(self.spatial_resolution)
        self.spatial_field = np.linspace(0, 1, self.spatial_resolution)

        # Initialize the log file
        if rank == 0:
            self.log_file = open("log_file.dat", "w")

    def compute_model_prediction(self, theta):
        """Function computes the model prediciton"""

        alpha, gamma, delta = self.extract_prameters(theta)

        T_prediction = []
        for isample in range(self.num_samples):
            T_inf = self.xtrain[isample]
            T_prediction.append(
                compute_prediction(
                    alpha=alpha,
                    gamma=gamma,
                    delta=delta,
                    T_inf=T_inf
                )
            )

        T_prediction = np.array(T_prediction).T

        return T_prediction

    def compute_log_likelihood(self, theta):
        """Function comptues the log likelihood (unnormalized)"""
        prediction = self.compute_model_prediction(theta)
        error = self.ytrain - prediction
        error_norm_sq = np.linalg.norm(error, axis=0)**2
        if self.model_noise_cov_scalar != 0:
            log_likelihood = -0.5 * \
                np.sum(error_norm_sq / self.model_noise_cov_scalar)
        else:
            log_likelihood = -0.5*np.sum(error_norm_sq)
        scaled_log_likelihood = self.objective_scaling*log_likelihood

        self.write_log_file("Scaled Log likelihood : {0:.5e} at theta : {1}".format(
            scaled_log_likelihood, theta))

        return scaled_log_likelihood

    def compute_mle(self):
        """Function computes the mle"""
        def objective_func(theta):
            return -self.compute_log_likelihood(theta)

        res = minimize(objective_func, np.random.randn(
            self.num_parameters), method="Nelder-Mead")
        res = minimize(objective_func, res.x)
        self.write_log_file("MLE computation finished!!!")
        self.write_log_file("Theta MLE : {} | Success : {}".format(res.x, res.success))

        return res.x

    def compute_emmisivity_distribution(self, theta):
        """Function computes the emmissivity"""
        alpha, gamma, delta = self.extract_prameters(theta)

        T = np.linspace(10, 100, 100)

        return 1e-4 * (gamma + delta*np.sin(alpha*T) + np.exp(0.02*T))

    def extract_prameters(self, theta):
        """Function extracts the parameters"""
        # alpha = theta[0]
        gamma = theta[1]
        delta = theta[2]

        alpha = 3*np.pi/200
        # gamma = 1
        # delta = 5

        return alpha, gamma, delta

    def write_log_file(self, message):
        """Function writes the log file"""
        if rank == 0:
            self.log_file.write(message+"\n")
            self.log_file.flush()
        else:
            pass

    def compute_estimated_mutual_information(self):
        """Function computes the estimated mutual information"""

        self.write_log_file("Begin Mutual information extimation")
        sample_based_approximator = approx_mutual_information(
            forward_model=self.compute_model_prediction,
            eval_mean=self.prior_mean,
            eval_cov=self.prior_cov,
            model_noise_cov_scalar=self.model_noise_cov_scalar,
            num_outer_samples=self.num_outer_samples,
            num_inner_samples=self.num_inner_samples,
            log_file=self.log_file if rank == 0 else None
        )

        estimated_mutual_information = sample_based_approximator.estimate_individual_mutual_information()

        if rank == 0:
            np.save("estimated_mutual_information.npy", estimated_mutual_information)

        self.write_log_file("End Mutual information estimation")

    def compute_estimated_conditional_mutual_information(self):
        """Function computes the estimated conditional mutual information"""

        self.write_log_file("Begin Conditional mutual information extimation")
        sample_based_approximator = approx_conditional_mutual_information(
            forward_model=self.compute_model_prediction,
            eval_mean=self.prior_mean,
            eval_cov=self.prior_cov,
            model_noise_cov_scalar=self.model_noise_cov_scalar,
            num_outer_samples=self.num_outer_samples,
            num_inner_samples=self.num_inner_samples,
            log_file=self.log_file if rank == 0 else None
        )

        estimated_conditional_mutual_information = sample_based_approximator.estimate_pair_mutual_infomation()

        if rank == 0:
            np.save("estimated_conditional_mutual_information.npy", estimated_conditional_mutual_information)

        self.write_log_file("End Conditional mutual information extimation")

    def update_prior(self, theta_mle):
        """Function updates the prior"""
        if rank == 0:
            update_prior = [theta_mle.reshape(-1, 1)]*size
        else:
            update_prior = None

        comm.Barrier()
        self.prior_mean = comm.scatter(update_prior, root=0)


def main():
    alpha_true = 3*np.pi/200
    gamma_true = 1
    delta_true = 5

    true_theta = np.array([alpha_true, gamma_true, delta_true])

    # Initialize the prior
    prior_mean = true_theta.copy().reshape(-1, 1)
    prior_cov = np.eye(prior_mean.shape[0])

    xtrain = np.array([50])
    model_noise_cov_scalar = 1e-1
    objective_scaling = 1e-10
    num_outer_samples = 100
    num_inner_samples = 100

    model = inference(
        xtrain=xtrain,
        model_noise_cov_scalar=model_noise_cov_scalar,
        true_theta=true_theta,
        objective_scaling=objective_scaling,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        num_outer_samples=num_outer_samples,
        num_inner_samples=num_inner_samples
    )

    # MLE
    # theta_mle = model.compute_mle()
    theta_mle = true_theta

    if rank == 0:
        np.save("theta_mle.npy", theta_mle)

    # Update the prior
    model.update_prior(theta_mle)

    # Model identifiability
    model.compute_estimated_mutual_information()
    model.compute_estimated_conditional_mutual_information()

    # if rank == 0:
    #     # Emmissivity
    #     true_emmisivity = model.compute_emmisivity_distribution(theta=true_theta)
    #     prediction_emmisivity = model.compute_emmisivity_distribution(theta=theta_mle)
    #     # Temperature distribution
    #     true_temperature = model.compute_model_prediction(theta=true_theta)
    #     prediction_temperature = model.compute_model_prediction(theta=theta_mle)
    #     fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    #     axs[0].scatter(model.spatial_field, model.ytrain.ravel(), c='k', s=30, label="Data")
    #     axs[0].plot(model.spatial_field, prediction_temperature.ravel(), lw=3, color='red', label="Prediction")
    #     axs[0].legend()
    #     axs[0].grid(axis="both")
    #     axs[0].set_xlabel("z")
    #     axs[0].set_ylabel("T(z)")
    #     axs[1].plot(np.linspace(10, 100, 100), np.abs(true_emmisivity - prediction_emmisivity), color="k", lw=3)
    #     axs[1].set_xlim(left=10, right=100)
    #     axs[1].grid(axis="both")
    #     axs[1].set_yscale("log")
    #     axs[1].set_xlabel("T")
    #     axs[1].set_ylabel(r"$|\epsilon_{true}-\epsilon_{prediction}|$")
    #     fig.suptitle(r"$T_{{\infty}} = {}$".format(50))
    #     plt.tight_layout()
    #     plt.savefig("prediction.png")
    #     plt.show()

    # Close the log file
    if rank == 0:
        model.log_file.close()



if __name__ == "__main__":
    main()
