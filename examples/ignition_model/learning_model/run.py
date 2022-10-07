import numpy as np
import time
import matplotlib.pyplot as plt
import yaml
from scipy.optimize import minimize
import sys
import time
import os
from mpi4py import MPI
sys.path.append("/home/sbhola/Documents/CASLAB/GIM/examples/ignition_model/forward_model/methane_combustion")
sys.path.append("/home/sbhola/Documents/CASLAB/GIM/information_metrics")
from combustion import mech_1S_CH4_Westbrook, mech_2S_CH4_Westbrook
from compute_identifiability import mutual_information, conditional_mutual_information

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=20)
plt.rc("lines", linewidth=2)
plt.rc("axes", labelpad=30, titlepad=20)

class learn_ignition_model():
    def __init__(self, config_data, campaign_path, prior_mean, prior_cov):
        """Atribute initialization"""
        # Prior
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.num_parameters = self.prior_mean.shape[0]

        # Extract configurations
        self.campaign_path = campaign_path
        self.equivalence_ratio = config_data["equivalence_ratio"]
        self.initial_temperature = config_data["initial_temperature"]
        self.initial_pressure = config_data["initial_pressure"]
        self.model_noise_cov = config_data["model_noise_cov"]
        assert(len(self.initial_temperature) == len(self.initial_pressure) == len(self.equivalence_ratio)), "Provide input config for all cases"

        # Traning data
        self.ytrain = np.load(os.path.join(self.campaign_path, "ytrain.npy"))
        self.spatial_resolution = self.ytrain.shape[1]
        self.num_data_points = len(self.initial_temperature)
        self.objective_scaling = config_data["objective_scaling"]

    def compute_Arrehenius_A(self, theta, equivalence_ratio, initial_temperature):
        """Function computes the pre exponential factor, A for the Arrhenius rate"""
        lambda_1 = 20 + (2)*theta[0] 
        lambda_2 = theta[1] 
        lambda_3 = theta[2] 
        log_A = lambda_1 + np.tanh((lambda_2 + lambda_3*equivalence_ratio)*(initial_temperature/1000))
        return np.exp(log_A)

    def compute_Arrehenius_Ea(self, theta):
        """Function computes the Arrhenius activation energy"""
        Arrhenius_Ea = 30000 + 10000*theta[3]
        return Arrhenius_Ea

    def compute_model_prediction(self, theta):
        """Function computes the model prediction, Temperature"""
        prediction = np.zeros((self.num_data_points, self.spatial_resolution))

        # Arrhenius_Ea = self.compute_Arrehenius_Ea(theta)

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

            prediction[idata_sample, :] = states.T

        return prediction

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
        print(res)
        print(self.compute_Arrehenius_A(res.x, self.equivalence_ratio[0], self.initial_temperature[0]))
        prediction_init = self.compute_model_prediction(theta=theta_init)
        prediction_mle = self.compute_model_prediction(theta=res.x)
        plt.figure()
        plt.plot(prediction_init.ravel(), label="Init")
        plt.plot(self.ytrain.ravel(), label="data")
        plt.plot(prediction_mle.ravel(), label="MLE")
        plt.legend()
        plt.show()
        return res.x

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

        print(res)
        print(self.compute_Arrehenius_A(res.x, self.equivalence_ratio[0], self.initial_temperature[0]))
        prediction_init = self.compute_model_prediction(theta=theta_init)
        prediction_mle = self.compute_model_prediction(theta=res.x)
        plt.figure()
        plt.plot(prediction_init.ravel(), label="Init")
        plt.plot(self.ytrain.ravel(), label="data")
        plt.plot(prediction_mle.ravel(), label="MAP")
        plt.legend()
        plt.show()

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
        learning_model.compute_mle()
    elif config_data['compute_map']:
        learning_model.compute_map()
    else:
        raise ValueError("Invalid selection")

if __name__=="__main__":
    main()



