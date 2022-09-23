import numpy as np
import matplotlib.pyplot as plt
import yaml
from scipy.optimize import minimize
import sys
sys.path.append("../forward_model/n_dodecane_ignition")
from n_dodecane_ignition import n_dodecane_combustion
plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=20)


class ignition_model():
    def __init__(self, model_inputs, true_theta, model_noise_cov, objective_scaling):
        self.model_inputs = model_inputs
        self.true_theta = true_theta
        self.model_noise_cov = model_noise_cov
        self.num_model_input_evals = self.model_inputs.shape[0]
        self.objective_scaling = objective_scaling

        noise = np.sqrt(self.model_noise_cov) * \
            np.random.randn(self.num_model_input_evals)
        self.ytrain = self.compute_model_prediction(
            theta=self.true_theta) + noise

    def compute_model_prediction(self, theta):
        """Function comptues the model prediction"""
        pre_exponential_parameter, activation_energy = self.extract_parameters(
            theta=theta)
        prediction = np.zeros(self.num_model_input_evals)

        for ii in range(self.num_model_input_evals):
            initial_temperature, equivalence_ratio = self.model_inputs[ii, :]
            ignition_time = n_dodecane_combustion(
                equivalence_ratio=equivalence_ratio,
                initial_temp=initial_temperature,
                pre_exponential_parameter=pre_exponential_parameter,
                activation_energy=activation_energy
            )
            prediction[ii] = np.log(ignition_time)
        return prediction

    def extract_parameters(self, theta):
        """Function extracts the parameters given the theta"""
        pre_exponential_parameter = theta[0]
        activation_energy = theta[1]
        return pre_exponential_parameter, activation_energy

    def compute_log_likelihood(self, theta):
        """Function comptuest the log likelihood"""
        prediction = self.compute_model_prediction(theta)
        error = self.ytrain - prediction
        error_norm_sq = np.linalg.norm(error)**2
        log_likelihood = -0.5*error_norm_sq/self.model_noise_cov
        return log_likelihood

    def compute_mle(self):
        """Function comptues the mle"""
        theta_init = np.array([27, 31000])

        def objective_function(theta):
            objective = -self.objective_scaling*self.compute_log_likelihood(theta)
            print(objective)
            return objective

        res = minimize(objective_function, theta_init,
                       method="Nelder-Mead")
        res = minimize(objective_function, res.x)
        print(res)
        return res.x

    def plot_prediction(self, theta):
        """Function plots the prediction"""
        prediction = self.compute_model_prediction(theta)
        plt.figure()
        plt.scatter(self.model_inputs[:, 0], self.ytrain, c="k", s=30, label="Data")
        plt.scatter(self.model_inputs[:, 0], prediction, c="r", s=30, label="Prediction")
        plt.legend()
        plt.xlabel(r"Temperature (K)")
        plt.ylabel(r"$\ln \Gamma$")
        plt.tight_layout()
        plt.savefig("prediction.png")
        plt.close()


def gather_model_inputs():
    """Function gathers the model inputs"""
    # Reading the config file
    with open("config.yaml", "r") as yaml_file:
        data = yaml.safe_load(yaml_file)
    temperature_config = data['temperature_inputs']
    equivalence_ratio_config = data['equivalence_ratio_inputs']
    true_theta = np.array(data["true_theta"])
    model_noise_cov = np.array(data["model_noise_cov"])
    load_theta_mle = data['load_theta_mle']
    objective_scaling = float(data['objective_scaling'])

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


    return inputs, true_theta, model_noise_cov, load_theta_mle, objective_scaling


def main():
    model_inputs, true_theta, model_noise_cov, load_theta_mle, objective_scaling = gather_model_inputs()

    model = ignition_model(
        model_inputs=model_inputs,
        true_theta=true_theta,
        model_noise_cov=model_noise_cov,
        objective_scaling=objective_scaling
    )

    if load_theta_mle:
        theta_mle = np.load("theta_mle.npy")
    else:
        theta_mle = model.compute_mle()
        np.save("theta_mle.npy", theta_mle)
    print(theta_mle)
    model.plot_prediction(theta_mle)


if __name__ == "__main__":
    main()
