# Author: Sahil Bhola
# Date: 6/6/2022
# Description: Identifiability ananlysis of the radiative heat transfer model
# True dynamics are considered and the parameters of the emmissivity are identified
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
import os
from mpi4py import MPI
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
sys.path.append("../forward_model")
sys.path.append("../../../information_metrics")

from rht_true import compute_prediction
from compute_identifiability_parallel_version import conditional_mutual_information

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
matplotlib.rcParams['font.family'] = 'serif'

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class inference():
    def __init__(self, xtrain, model_noise_cov_scalar, true_theta, objective_scaling, prior_mean, prior_cov, num_outer_samples, num_inner_samples, loaded_ytrain=None, restart=False):
        self.xtrain = xtrain
        self.model_noise_cov_scalar = model_noise_cov_scalar
        self.num_data_samples = self.xtrain.shape[0]
        self.true_theta = true_theta
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.num_parameters = self.true_theta.shape[0]
        self.objective_scaling = objective_scaling
        self.num_outer_samples = num_outer_samples
        self.num_inner_samples = num_inner_samples
        self.restart = restart

        if loaded_ytrain is None:
            self.ytrain = self.compute_model_prediction(theta=self.true_theta)
            self.spatial_res = self.ytrain.shape[1]

            noise = np.sqrt(self.model_noise_cov_scalar)*np.random.randn(
                self.num_data_samples*(self.spatial_res-2)).reshape(self.num_data_samples, -1)
            self.ytrain[:, 1:-1] += noise
            if rank == 0:
                np.save("ytrain.npy", self.ytrain)

        else:
            self.ytrain = loaded_ytrain
            self.spatial_res = self.ytrain.shape[1]

        self.model_noise_cov_mat = self.model_noise_cov_scalar * \
            np.eye(self.spatial_res)
        self.spatial_field = np.linspace(0, 1, self.spatial_res)

        # Initialize the log file
        if rank == 0:
            self.log_file = open("log_file.dat", "w")
        else:
            self.log_file = None

    def compute_model_prediction(self, theta):
        """Function computes the model prediciton"""

        alpha, gamma, delta = self.extract_prameters(theta)

        T_prediction = []
        for isample in range(self.num_data_samples):
            T_inf = self.xtrain[isample]
            T_prediction.append(
                compute_prediction(
                    alpha=alpha,
                    gamma=gamma,
                    delta=delta,
                    T_inf=T_inf
                )
            )

        T_prediction = np.array(T_prediction)

        return T_prediction

    def compute_log_likelihood(self, theta):
        """Function comptues the log likelihood (unnormalized)"""
        prediction = self.compute_model_prediction(theta)
        error = self.ytrain - prediction
        error_norm_sq = np.linalg.norm(error, axis=1)**2
        if self.model_noise_cov_scalar != 0:
            log_likelihood = -0.5 * \
                np.sum(error_norm_sq / self.model_noise_cov_scalar, axis=0)
        else:
            log_likelihood = -0.5*np.sum(error_norm_sq, axis=0)
        scaled_log_likelihood = self.objective_scaling*log_likelihood

        self.write_log_file("Scaled Log likelihood : {0:.5e} at theta : {1}".format(
            scaled_log_likelihood, theta))

        return scaled_log_likelihood

    def compute_mle(self):
        """Function computes the mle"""
        def objective_func(theta):
            return -self.compute_log_likelihood(theta)

        res = minimize(objective_func, np.random.randn(
            self.num_parameters), method="Nelder-Mead", options={'maxiter':500})
        res = minimize(objective_func, res.x)
        self.write_log_file("MLE computation finished!!!")
        self.write_log_file("Theta MLE : {} | Success : {}".format(res.x, res.success))
        comm.Barrier()
        self.write_log_file("Converged at all procs.")

        return res.x

    def compute_emmisivity_distribution(self, theta):
        """Function computes the emmissivity"""
        alpha, gamma, delta = self.extract_prameters(theta)

        T = np.linspace(20, 100, 200)

        return 1e-4 * (gamma + delta*np.sin(alpha*T) + np.exp(0.02*T))

    def extract_prameters(self, theta):
        """Function extracts the parameters"""
        alpha = theta[0]
        gamma = theta[1]
        # delta = theta[2]

        # alpha = 3*np.pi/200
        # gamma = 1
        delta = 5

        return alpha, gamma, delta

    def write_log_file(self, message):
        """Function writes the log file"""
        if rank == 0:
            self.log_file.write(message+"\n")
            self.log_file.flush()
        else:
            pass

    def estimate_parameter_conditional_mutual_information(self):
        """Function computes the individual mutual information, I(theta_i;Y|theta_[rest of parameters])"""
        estimator = conditional_mutual_information(
                forward_model=self.compute_model_prediction,
                prior_mean=self.prior_mean,
                prior_cov=self.prior_cov,
                model_noise_cov_scalar=self.model_noise_cov_scalar,
                global_num_outer_samples=self.num_outer_samples,
                global_num_inner_samples=self.num_inner_samples,
                ytrain=self.ytrain,
                save_path=os.getcwd(),
                log_file=self.log_file,
                restart=self.restart
                )
        breakpoint()

        estimator.compute_individual_parameter_data_mutual_information_via_mc(
                use_quadrature=True,
                single_integral_gaussian_quad_pts=60
                )
        estimator.compute_posterior_pair_parameter_mutual_information(
                use_quadrature=True,
                single_integral_gaussian_quad_pts=60,
                double_integral_gaussian_quad_pts=60
                )

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
    # loaded_ytrain = None
    # theta_mle = None
    loaded_ytrain = np.load('ytrain.npy')
    theta_mle = np.load('theta_mle.npy')
    restart = False

    # true_theta = np.array([alpha_true, gamma_true, delta_true])
    true_theta = np.array([alpha_true, gamma_true])

    # Initialize the prior
    prior_mean = true_theta.copy().reshape(-1, 1)
    prior_cov = np.eye(prior_mean.shape[0])

    xtrain = np.array([50])
    model_noise_cov_scalar = 1e+1
    objective_scaling = 1e-10
    num_outer_samples = 500
    num_inner_samples = 20

    model = inference(
        xtrain=xtrain,
        model_noise_cov_scalar=model_noise_cov_scalar,
        true_theta=true_theta,
        objective_scaling=objective_scaling,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        num_outer_samples=num_outer_samples,
        num_inner_samples=num_inner_samples,
        loaded_ytrain = loaded_ytrain,
        restart=restart
    )

    # MLE
    if theta_mle is None:
        theta_mle = model.compute_mle()

        if rank == 0:
            np.save("theta_mle.npy", theta_mle)
    # Update the prior
    model.update_prior(theta_mle)

    # Model identifiability
    model.estimate_parameter_conditional_mutual_information()


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
    #     axs[1].plot(np.linspace(20, 100, 200), np.abs(true_emmisivity - prediction_emmisivity), color="k", lw=3)
    #     axs[1].set_xlim(left=20, right=100)
    #     axs[1].grid(axis="both")
    #     axs[1].set_yscale("log")
    #     axs[1].set_xlabel("T")
    #     axs[1].set_ylabel(r"$|\epsilon_{true}-\epsilon_{prediction}|$")
    #     axs[1].xaxis.set_minor_locator(MultipleLocator(5))
    #     fig.suptitle(r"$T_{{\infty}} = {}$".format(50))
    #     plt.tight_layout()
    #     plt.savefig("prediction.png")
    #     plt.show()

    # Close the log file
    if rank == 0:
        model.log_file.close()



if __name__ == "__main__":
    main()
