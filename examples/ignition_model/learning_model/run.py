import numpy as np
import matplotlib.pyplot as plt
import yaml
from scipy.optimize import minimize
import sys
import os
from mpi4py import MPI
sys.path.append("../forward_model/n_dodecane_ignition")
from n_dodecane_ignition import n_dodecane_combustion

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=20)


class ignition_model():
    def __init__(self, model_inputs, true_theta, model_noise_cov, objective_scaling, prior_mean, prior_cov, ytrain, campaign_path):
        self.model_inputs = model_inputs
        self.true_theta = true_theta
        self.model_noise_cov = model_noise_cov
        self.num_model_input_evals = self.model_inputs.shape[0]
        self.objective_scaling = objective_scaling
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.campaign_path = campaign_path

        if rank == 0:
            log_file_name = os.path.join(self.campaign_path, "log_file.dat")
            self.log_file = open(log_file_name, "w")
        else:
            self.log_file = None

        noise = np.sqrt(self.model_noise_cov) * \
            np.random.randn(self.num_model_input_evals).reshape(self.num_model_input_evals, 1)
        if ytrain is None:
            self.ytrain = self.compute_model_prediction(
                theta=self.true_theta) + noise
            ytrain_file_name = "ytrain_without_sub_sampling_noise_{}.npy".format(self.model_noise_cov)
            save_path = os.path.join(self.campaign_path, ytrain_file_name)
            np.save(save_path, self.ytrain)
        else:
            self.ytrain = ytrain

        self.write_log_file("Data procured")
        breakpoint()

    def compute_model_prediction(self, theta):
        """Function comptues the model prediction"""
        pre_exponential_parameter, activation_energy = self.extract_parameters(
            theta=theta)
        prediction = np.zeros((self.num_model_input_evals, 1))

        for ii in range(self.num_model_input_evals):
            initial_temperature, equivalence_ratio = self.model_inputs[ii, :]
            ignition_time = n_dodecane_combustion(
                equivalence_ratio=equivalence_ratio,
                initial_temp=initial_temperature,
                pre_exponential_parameter=pre_exponential_parameter,
                activation_energy=activation_energy,
                campaign_path=self.campaign_path
            )
            prediction[ii, 0] = np.log(ignition_time)
        return prediction

    def extract_parameters(self, theta):
        """Function extracts the parameters given the theta"""
        pre_exponential_parameter = theta[0]*5 + 25
        activation_energy = theta[1]*5000 + 35000
        # print('pre_exponential_parameter {}'.format(pre_exponential_parameter))
        # print('activation_energy : {}'.format(activation_energy))
        # pre_exponential_parameter = theta[0]
        # activation_energy = theta[1]
        return pre_exponential_parameter, activation_energy

    def compute_log_likelihood(self, theta):
        """Function comptuest the log likelihood"""
        prediction = self.compute_model_prediction(theta)
        error = self.ytrain - prediction
        error_norm_sq = np.linalg.norm(error)**2
        log_likelihood = -0.5*error_norm_sq/self.model_noise_cov
        return log_likelihood

    def compute_log_prior(self, theta):
        """Function computes the log prior"""
        error = (theta - self.prior_mean.ravel()).reshape(-1, 1)
        exp_term_solve = self.prior_mean.T@np.linalg.solve(self.prior_cov, error)
        log_prior =  -0.5*exp_term_solve
        return log_prior.item()

    def compute_unnormalized_posterior(self, theta):
        """Function computes the unnormalized posterior"""
        log_likelihood = self.compute_log_likelihood(theta)
        log_prior = self.compute_log_prior(theta)
        log_posterior = log_likelihood + log_prior
        return log_posterior

    def compute_mle(self, theta_init):
        """Function comptues the mle"""
        def objective_function(theta):
            objective = -self.objective_scaling*self.compute_log_likelihood(theta)
            self.write_log_file("MLE objective : {0:.6e}".format(objective))
            return objective
        res = minimize(objective_function, theta_init,
                       method="Nelder-Mead")
        res = minimize(objective_function, res.x)
        if res.sucess is False:
            model.write_log_file("Numerical minima not found")

        self.write_log_file("MLE computation finished")

        return res.x

    def compute_map(self, theta_init):
        """Function comptues the map estimate"""
        def objective_function(theta):
                    objective = -self.objective_scaling*self.compute_unnormalized_posterior(theta)
                    self.write_log_file("MAP objective : {0:.6e}".format(objective))
                    return objective

        res = minimize(objective_function, theta_init,
                       method="Nelder-Mead")
        res = minimize(objective_function, res.x)

        if res.sucess is False:
            model.write_log_file("Numerical minima not found")

        self.write_log_file("MAP computation finished")

        theta_map = res.x
        cov_approx = res.hess_inv

        return theta_map, cov_approx

    def parameter_prior(self, updated_mean, updated_cov):
        """Function updates the parameter prior"""
        self.prior_mean = updated_mean
        self.prior_cov = updated_cov

    def plot_prediction(self, theta):
        """Function plots the prediction"""
        prediction = self.compute_model_prediction(theta)
        plt.figure()
        plt.scatter(self.model_inputs[:, 0], self.ytrain.ravel(), c="k", s=30, label="Data")
        plt.scatter(self.model_inputs[:, 0], prediction.ravel(), c="r", s=30, label="Prediction")
        plt.legend()
        plt.xlabel(r"Temperature (K)")
        plt.ylabel(r"$\ln \Gamma$")
        plt.tight_layout()
        plt.savefig("prediction.png")
        plt.close()

    def write_log_file(self, message):
        """Function writes the message on the log file"""
        if rank == 0:
            assert(self.log_file is not None), "log file must be provided"
            self.log_file.write(message+"\n")
            self.log_file.flush()


def load_configuration_file(file_name="config.yaml"):
    """Function loads the configuration file"""
    with open(file_name, 'r') as config:
        config_data = yaml.safe_load(config)
    return config_data

def generate_model_inputs(config_data):
    """Function gathers the model inputs"""
    # Reading the config file
    temperature_config = config_data['temperature_inputs']
    equivalence_ratio_config = config_data['equivalence_ratio_inputs']

    num_temp_points = len(temperature_config)
    num_equivalence_points = len(equivalence_ratio_config)

    if num_temp_points == num_equivalence_points:
        temperature = np.array(temperature_config)
        equivalence_ratio = np.array(equivalence_ratio_config)
    else:
        if (num_equivalence_points == 1):
            print("Using the same equivalence ratio for all temperature inputs")
            equivalence_ratio = np.array(
                equivalence_ratio_config*num_temp_points)
            temperature = np.array(temperature_config)
        elif (num_temp_points == 1):
            print("Using same temperature inputs for all the equivalence ratios")
            temperature = np.array(temperature_config*num_equivalence_points)
            equivalence_ratio = np.array(equivalence_ratio_config)
        else:
            raise ValueError(
                "Invalid number of temperature and equivalence ratio inputs")

    inputs = np.zeros((max(num_equivalence_points, num_temp_points), 2))
    inputs[:, 0] = temperature
    inputs[:, 1] = equivalence_ratio


    return inputs

def main():
    # Input configuration
    config_data = load_configuration_file()
    campaign_id = config_data['campaign_id']
    model_inputs = generate_model_inputs(config_data)
    true_theta = np.array(config_data['true_theta'])
    model_noise_cov = config_data['model_noise_cov']
    objective_scaling = config_data['objective_scaling']
    compute_mle = config_data['compute_mle']
    compute_map = config_data['compute_map']

    load_parameter_estimate = config_data['load_parameter_estimate']
    load_ytrain = config_data['load_ytrain']

    campaign_path = os.path.join(os.getcwd(), "campaign_results/campaign_{}".format(campaign_id))

    # Load the data
    if load_ytrain:
        ytrain_file_name = "ytrain_without_sub_sampling_noise_{}.npy".format(model_noise_cov)
        ytrain = np.load(os.path.join(campaign_path, ytrain_file_name))
    else:
        ytrain = None

    # Load the parameters
    if load_parameter_estimate:
        parameter_mean_file_name = "theta_mean_noise_{}.npy".format(model_noise_cov)
        parameter_cov_file_name = "theta_cov_noise_{}.npy".format(model_noise_cov)
        prior_mean = np.load(os.path.join(campaign_path, parameter_mean_file_name))
        prior_cov = np.load(os.path.join(campaign_path, parameter_cov_file_name))
    else:
        prior_mean = np.zeros(true_theta.shape[0]).reshape(-1, 1)
        prior_cov = np.eye(true_theta.shape[0])


    model = ignition_model(
        model_inputs=model_inputs,
        true_theta=true_theta,
        model_noise_cov=model_noise_cov,
        objective_scaling=objective_scaling,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        ytrain=ytrain,
        campaign_path=campaign_path
    )

    if load_parameter_estimate is False:
        if compute_mle:
            theta_mle = model.compute_mle(theta_init=prior_mean.ravel())
            updated_mean = theta_mle.reshape(-1, 1)
            updated_cov = np.eye(true_theta.shape[0])
        elif compute_map:
            theta_map, cov_map = model.compute_map(theta_init=prior_mean.ravel())
            updated_mean = theta_map
            updated_cov = cov_map
        else:
            raise ValueError("Select an estimator type")

        parameter_mean_file_name = "theta_mean_noise_{}.npy".format(model_noise_cov)
        parameter_cov_file_name = "theta_cov_noise_{}.npy".format(model_noise_cov)

        np.save(os.path.join(campaign_path, parameter_mean_file_name), updated_mean)
        np.save(os.path.join(campaign_path, parameter_cov_file_name), updated_cov)

        # Update model prior
        model.parameter_prior(updated_mean=updated_mean, updated_cov=updated_cov)
    model.plot_prediction(theta=prior_mean)

    # Close the write file
    if rank == 0:
        model.log_file.close()

if __name__ == "__main__":
    main()
