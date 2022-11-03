import numpy as np
import time
import matplotlib.pyplot as plt
import pylab
import yaml
from scipy.optimize import minimize
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import sys
import time
import os
from mpi4py import MPI
sys.path.append("../../forward_model/methane_combustion")
sys.path.append("../../../../information_metrics")
from combustion import mech_1S_CH4_Westbrook, mech_2S_CH4_Westbrook, mech_gri 
from compute_identifiability import mutual_information, conditional_mutual_information

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=25)
plt.rc("lines", linewidth=3)
plt.rc("axes", labelpad=30, titlepad=20)
plt.rc("legend", fontsize=18, framealpha=1.0)
plt.rc("xtick.major", size=6)
plt.rc("xtick.minor", size=4)
plt.rc("ytick.major", size=6)
plt.rc("ytick.minor", size=4)

class learn_ignition_model():
    def __init__(self, config_data, campaign_path, prior_mean, prior_cov):
        """Atribute initialization"""
        # Prior
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.num_parameters = self.prior_mean.shape[0]

        # Extract configurations
        self.campaign_path = campaign_path
        self.model_noise_cov = config_data["model_noise_cov"]

        # Traning data
        training_data_set = np.load(os.path.join(self.campaign_path, "training_data.npy"))
        self.initial_temperature = training_data_set[:, 0]
        self.initial_pressure = training_data_set[:, 1]
        self.equivalence_ratio = training_data_set[:, 2]
        self.ytrain = training_data_set[:, -1].reshape(-1, 1)

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

    def compute_Arrehenius_A(self, theta, equivalence_ratio, initial_temperature):
        """Function computes the pre exponential factor, A for the Arrhenius rate"""
        lambda_1 = 18 + (1)*theta[0] 
        lambda_2 = theta[1] 
        lambda_3 = theta[2] 
        log_A = lambda_1 + np.tanh((lambda_2 + lambda_3*equivalence_ratio)*(initial_temperature/1000))
        return np.exp(log_A)

    def compute_Arrehenius_Ea(self, theta):
        """Function computes the Arrhenius activation energy"""
        Arrhenius_Ea = 30000 + 10000*theta[3]
        return Arrhenius_Ea

    def compute_model_prediction(self, theta, proc_log_file=None):
        """Function computes the model prediction, Temperature"""
        prediction = np.zeros((self.num_data_points, self.spatial_resolution))
        for idata_sample in range(self.num_data_points):
            initial_pressure = self.initial_pressure[idata_sample]
            initial_temperature = self.initial_temperature[idata_sample]
            equivalence_ratio = self.equivalence_ratio[idata_sample]
            
            # Compute the Arrehnius rate
            Arrhenius_A = self.compute_Arrehenius_A(
                    theta=theta,
                    equivalence_ratio=equivalence_ratio,
                    initial_temperature=initial_temperature
                    )
            if proc_log_file is not None:
                proc_log_file.write("     Arrhenius_A: {0:.18f}\n".format(Arrhenius_A))
                proc_log_file.flush()

            combustion_model = mech_2S_CH4_Westbrook(
                    initial_temperature=initial_temperature,
                    initial_pressure=initial_pressure,
                    equivalence_ratio=equivalence_ratio,
                    Arrhenius_A=Arrhenius_A,
                    )
            ignition_time = combustion_model.compute_ignition_time()
            prediction[idata_sample, :] = np.log(ignition_time) 
            if proc_log_file is not None:
                proc_log_file.write("     Prediction: {}\n".format(prediction[idata_sample, :]))
                proc_log_file.flush()

        return prediction

    def compute_complete_state(self, theta):
        """Function computes the entire state"""
        state_list = []
        for idata_sample in range(self.num_data_points):
            Arrhenius_A = self.compute_Arrehenius_A(
                    theta=theta,
                    equivalence_ratio=self.equivalence_ratio[idata_sample],
                    initial_temperature=self.initial_temperature[idata_sample]
                    )
            # print(Arrhenius_A, Arrhenius_Ea)
            combustion_model = mech_1S_CH4_Westbrook(
                    initial_temperature=self.initial_temperature[idata_sample],
                    initial_pressure=self.initial_pressure[idata_sample],
                    equivalence_ratio=self.equivalence_ratio[idata_sample],
                    Arrhenius_A=Arrhenius_A,
                    # Arrhenius_Ea=Arrhenius_Ea,
                    )

            states = combustion_model.mech_1S_CH4_Westbrook_combustion()

            state_list.append(states)
        
        return state_list

    def compute_log_likelihood(self, theta):
        """Function comptues the log likelihood"""
        prediction = self.compute_model_prediction(theta=theta)
        error = self.ytrain - prediction
        error_norm_sq = np.linalg.norm(error, axis=1)**2
        log_likelihood = (-0.5/self.model_noise_cov)*(np.sum(error_norm_sq))
        return log_likelihood

    def compute_log_prior(self, theta):
        """Function computes the log prior"""
        error = (theta - self.prior_mean).reshape(-1, 1)
        error_norm_sq = (error.T@np.linalg.solve(self.prior_cov, error)).item()
        log_prior = -0.5*error_norm_sq
        return log_prior

    def compute_mle(self):
        """Function comptues the MLE"""
        theta_init = np.random.randn(self.num_parameters)
        def objective_function(theta):
            objective_function = - self.objective_scaling*self.compute_log_likelihood(theta)
            print("MLE objective : {0:.18e}".format(objective_function))
            return objective_function

        res = minimize(objective_function, theta_init, method="Nelder-Mead")
        res = minimize(objective_function, res.x)
        theta_mle = res.x

        prediction_mle = self.compute_model_prediction(theta=res.x).ravel()
        prediction_save = np.zeros((self.num_data_points, 4))
        prediction_save[:, 0] = self.initial_temperature
        prediction_save[:, 1] = self.initial_pressure
        prediction_save[:, 2] = self.equivalence_ratio
        prediction_save[:, 3] = prediction_mle 
        
        # Saving MLE
        save_prediction_path = os.path.join(self.campaign_path, "prediction_mle.dat")
        save_mle_path = os.path.join(self.campaign_path, "theta_mle.npy")

        np.savetxt(save_prediction_path, prediction_save, delimiter=' ', header = "Vartiables: Inital_temperature, Initial_pressure, Equivalence_ratio, Ignition_temperature")
        np.save(save_mle_path, theta_mle)

        return theta_mle

    def compute_unnormalized_posterior(self, theta):
        log_likelihood = self.compute_log_likelihood(theta)
        log_prior = self.compute_log_prior(theta)
        unnormalized_log_post = log_likelihood + log_prior

        return unnormalized_log_post

    def compute_map(self):
        """Function computes the map estimate"""
        theta_init = np.random.randn(self.num_parameters)
        
        def objective_function(theta):
            objective_function = - self.objective_scaling*self.compute_unnormalized_posterior(theta)
            print("MAP objective : {0:.18e}".format(objective_function))
            return objective_function

        res = minimize(objective_function, theta_init, method="Nelder-Mead")
        res = minimize(objective_function, res.x)

        theta_map = res.x
        theta_map_cov = res.hess_inv

        return theta_map.reshape(-1, 1), theta_map_cov

    def plot_map_estimate(self, theta_map, theta_map_cov):
        """Function plots the map estimate"""
        num_samples = 50
        temperature_range = np.linspace(833, 2333, num_samples) 
        calibrated_model_prediction = np.zeros((num_samples, temperature_range.shape[0]))
        un_calibrated_model_prediction = np.zeros_like(temperature_range)
        gri_prediction = np.zeros_like(temperature_range)

        d = theta_map.shape[0]
        std_normal_samples = np.random.randn(d, num_samples)
        cov_cholesky = np.linalg.cholesky(theta_map_cov)
        theta_samples = theta_map + cov_cholesky@std_normal_samples

        for ii in range(num_samples):
            print("sample : {}".format(ii))
            eval_theta = theta_samples[:, ii]
            for jj, itemp in enumerate(temperature_range):

                Arrhenius_A = self.compute_Arrehenius_A(
                        theta=eval_theta,
                        equivalence_ratio=1.0,
                        initial_temperature=itemp
                        )

                model_2S_CH4_Westbrook = mech_2S_CH4_Westbrook(
                        initial_temperature=itemp,
                        initial_pressure=100000,
                        equivalence_ratio = 1.0,
                        Arrhenius_A=Arrhenius_A
                        )

                calibrated_model_prediction[ii, jj] = np.log(model_2S_CH4_Westbrook.compute_ignition_time(internal_state="plot"))

        for ii,itemp in enumerate(temperature_range):
            # Compute gri prediciton
            gri_model = mech_gri(
                    initial_temperature=itemp,
                    initial_pressure=100000,
                    equivalence_ratio = 1.0
                    )
            gri_prediction[ii] = np.log(gri_model.compute_ignition_time())

            # Compute 2-step Westbrook prediction (uncalibrated)
            model_2S_CH4_Westbrook = mech_2S_CH4_Westbrook(
                    initial_temperature=itemp,
                    initial_pressure=100000,
                    equivalence_ratio = 1.0,
                    )

            un_calibrated_model_prediction[ii] = np.log(model_2S_CH4_Westbrook.compute_ignition_time(internal_state="plot"))


        mean_val = np.mean(calibrated_model_prediction, axis=0)
        std_val = np.std(calibrated_model_prediction, axis=0)

        save_fig_path = os.path.join(self.campaign_path, "Figures/prediction_map_phi_{}_pressure_{}_noise_{}.png".format(1.0, 100000, self.model_noise_cov))
        fig, axs = plt.subplots(figsize=(12, 7))
        axs.scatter(1000/self.initial_temperature, self.ytrain.ravel(), label="Data", color="k", marker="s", s=100)
        axs.plot(1000/temperature_range, gri_prediction, "--", label="Gri-Mech 3.0", color="k")
        axs.plot(1000/temperature_range, un_calibrated_model_prediction, label=r"2-step mechanism [Westbrook \textit{et al.}]", color="r")
        axs.plot(1000/temperature_range, mean_val, label=r"$\mu_{MAP}$", color="b")
        axs.fill_between(1000/temperature_range, mean_val+std_val, mean_val-std_val, ls="--", edgecolor="C0",lw=2, alpha=0.3, label=r"$\pm\sigma_{MAP}$")
        axs.set_xlabel(r"1000/T [1/K]")
        axs.set_ylabel(r"$\log(t_{ign})$")
        axs.legend(loc="lower right")
        axs.grid(True, axis="both", which="major",color="k", alpha=0.5)
        axs.grid(True, axis="both", which="minor",color="grey", alpha=0.3)
        axs.xaxis.set_minor_locator(MultipleLocator(0.05))
        axs.xaxis.set_major_locator(MultipleLocator(0.2))
        axs.yaxis.set_minor_locator(AutoMinorLocator())
        axs.set_xlim([0.2, 1.4])
        plt.tight_layout()
        plt.savefig(save_fig_path)
        plt.close()

    def plot_mle_estimate(self, theta_mle):
        """Function plots the mle estimate"""
        temperature_range = np.linspace(700, 3333, 50) 
        calibrated_model_prediction = np.zeros_like(temperature_range)
        un_calibrated_model_prediction = np.zeros_like(temperature_range)
        gri_prediction = np.zeros_like(temperature_range)

        for ii,itemp in enumerate(temperature_range):
            # Compute gri prediciton
            gri_model = mech_gri(
                    initial_temperature=itemp,
                    initial_pressure=100000,
                    equivalence_ratio = 1.0
                    )
            gri_prediction[ii] = gri_model.compute_ignition_time()

            # Compute 2-step Westbrook prediction (Calibrated)
            Arrhenius_A = self.compute_Arrehenius_A(
                    theta=theta_mle,
                    equivalence_ratio=1.0,
                    initial_temperature=itemp
                    )

            model_2S_CH4_Westbrook = mech_2S_CH4_Westbrook(
                    initial_temperature=itemp,
                    initial_pressure=100000,
                    equivalence_ratio = 1.0,
                    Arrhenius_A=Arrhenius_A
                    )

            calibrated_model_prediction[ii] = model_2S_CH4_Westbrook.compute_ignition_time()


            # Compute 2-step Westbrook prediction (uncalibrated)
            model_2S_CH4_Westbrook = mech_2S_CH4_Westbrook(
                    initial_temperature=itemp,
                    initial_pressure=100000,
                    equivalence_ratio = 1.0,
                    )

            un_calibrated_model_prediction[ii] = model_2S_CH4_Westbrook.compute_ignition_time()

        save_fig_path = os.path.join(self.campaign_path, "Figures/prediction_mle_phi_{}_pressure_{}_noise_{}.png".format(1.0, 100000, self.model_noise_cov))
        fig, axs = plt.subplots(figsize=(12, 7))
        axs.scatter(1000/self.initial_temperature, np.exp(self.ytrain.ravel()), label="Data", color="k", marker="s", s=100)
        axs.plot(1000/temperature_range, gri_prediction, "--", label="Gri-Mech 3.0", color="k")
        axs.plot(1000/temperature_range, calibrated_model_prediction, label="Modified 2-step mechanism", color="b")
        axs.plot(1000/temperature_range, un_calibrated_model_prediction, label=r"2-step mechanism [Westbrook \textit{et al.}]", color="r")
        axs.set_xlabel(r"1000/T [1/K]")
        axs.set_ylabel(r"$t_{ign}$ [s]")
        axs.legend(loc="lower right")
        axs.grid(True, axis="both", which="major",color="k", alpha=0.5)
        axs.grid(True, axis="both", which="minor",color="grey", alpha=0.3)
        axs.xaxis.set_minor_locator(MultipleLocator(0.05))
        axs.xaxis.set_major_locator(MultipleLocator(0.2))
        axs.yaxis.set_minor_locator(AutoMinorLocator())
        # axs.set_ylim([10**(-7), 10**(3)])
        axs.set_xlim([0.2, 1.6])
        axs.set_yscale("log")
        plt.tight_layout()
        plt.savefig(save_fig_path)
        plt.close()

    def compute_model_identifiability(self, prior_mean, prior_cov):
        """Function computes the model identifiability"""
        def forward_model(theta, proc_log_file=None):
            prediction = self.compute_model_prediction(theta, proc_log_file=proc_log_file).T
            return prediction

        estimator = conditional_mutual_information(
                forward_model=forward_model,
                prior_mean=prior_mean,
                prior_cov=prior_cov,
                model_noise_cov_scalar=self.model_noise_cov,
                global_num_outer_samples=self.global_num_outer_samples,
                global_num_inner_samples=self.global_num_inner_samples,
                save_path=self.campaign_path,
                restart=self.restart_identifiability,
                ytrain=self.ytrain.T,
                log_file=self.log_file,
                ) 
        
        estimator.compute_individual_parameter_data_mutual_information_via_mc(
                use_quadrature=True,
                single_integral_gaussian_quad_pts=7
                )

        estimator.compute_posterior_pair_parameter_mutual_information(
                use_quadrature=True,
                single_integral_gaussian_quad_pts=7,
                double_integral_gaussian_quad_pts=10
                )

def load_configuration_file(config_file_path="./config.yaml"):
    """Function loads the configuration file"""
    with open(config_file_path, "r") as config_file:
        config_data = yaml.safe_load(config_file)
    print("Loaded %s configuration file"%(config_file_path))
    return config_data

def main():
    # Begin user input
    prior_mean = np.zeros(3)
    prior_cov = np.eye(3)
    # End user input

    # Load the configurations
    config_data = load_configuration_file() 

    # Campaign path
    campaign_path = os.path.join(os.getcwd(), "campaign_results/campaign_%d"%(config_data["campaign_id"]))



    # Learning model
    learning_model = learn_ignition_model(
            config_data=config_data,
            campaign_path=campaign_path,
            prior_mean=prior_mean,
            prior_cov=prior_cov
            )

    if config_data['compute_mle']:
        theta_mle = learning_model.compute_mle()
        np.save(os.path.join(campaign_path, "theta_mle.npy"), theta_mle)
    elif config_data['compute_map']:
        theta_map, theta_map_cov = learning_model.compute_map()
        np.save(os.path.join(campaign_path, "theta_map.npy"), theta_map)
        np.save(os.path.join(campaign_path, "theta_map_cov.npy"), theta_map_cov)

    if ("--plotmap" in sys.argv):
        theta_map = np.load(os.path.join(campaign_path, "theta_map.npy"))
        theta_map_cov = np.load(os.path.join(campaign_path, "theta_map_cov.npy"))
        learning_model.plot_map_estimate(
                theta_map=theta_map,
                theta_map_cov=theta_map_cov
                )

    elif ("--plotmle" in sys.argv):
        theta_mle = np.load(os.path.join(campaign_path, "theta_mle.npy"))
        learning_model.plot_mle_estimate(
                theta_mle=theta_mle,
                )

    if config_data["compute_identifiability"]:
        theta_map = np.load(os.path.join(campaign_path, "theta_map.npy"))
        theta_map_cov = np.load(os.path.join(campaign_path, "theta_map_cov.npy"))
        learning_model.compute_model_identifiability(
                prior_mean=theta_map,
                prior_cov=theta_map_cov
                )

    if rank == 0:
        learning_model.log_file.close()

if __name__=="__main__":
    main()



