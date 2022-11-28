import numpy as np
import time
import matplotlib.pyplot as plt
import pylab
import yaml
from scipy.optimize import minimize
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import sys
import time
import os
from mpi4py import MPI

sys.path.append("../forward_model/methane_combustion")
sys.path.append("../../../information_metrics")
sys.path.append("../../../mcmc")
from combustion import mech_1S_CH4_Westbrook, mech_2S_CH4_Westbrook, mech_gri
from compute_identifiability import mutual_information, conditional_mutual_information
from mcmc import *
from mcmc_utils import *
from color_schemes import dark_colors

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=25)
plt.rc("lines", linewidth=3)
plt.rc("axes", labelpad=20, titlepad=20)
plt.rc("legend", fontsize=18, framealpha=1.0)
plt.rc("xtick.major", size=6)
plt.rc("xtick.minor", size=4)
plt.rc("ytick.major", size=6)
plt.rc("ytick.minor", size=4)


class learn_ignition_model:
    def __init__(self, config_data, campaign_path, prior_mean, prior_cov):
        """Atribute initialization"""
        # Prior
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.num_parameters = self.prior_mean.shape[0]

        # Extract configurations
        self.campaign_path = campaign_path
        self.model_noise_cov = config_data["model_noise_cov"]
        self.n_parameter_model = config_data["n_parameter_model"]

        # Traning data
        training_data_set = np.load(
            os.path.join(self.campaign_path, "training_data.npy")
        )
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

    def compute_kinetic_parameters(self, theta, initial_temperature, equivalence_ratio):
        """Function computes the kinetic parameters from the input parameters"""
        kinetic_parameters = {}
        kinetic_parameters["A"] = self.compute_Arrehenius_A(
            theta, initial_temperature, equivalence_ratio
        )
        kinetic_parameters["Ea"] = self.compute_Arrehenius_Ea(theta)
        return kinetic_parameters

    def compute_Arrehenius_A(self, theta, initial_temperature, equivalence_ratio):
        """Function computes the pre exponential factor, A for the Arrhenius rate"""

        if self.n_parameter_model == 4:
            l_0 = 18 + theta[0]
            l_1 = theta[1]
            l_2 = theta[2]
            log_A = l_0 + np.tanh(
                (l_1 + (l_2 * equivalence_ratio)) * (initial_temperature / 1000)
            )

        elif self.n_parameter_model == 8:
            l_0 = 18 + theta[0]
            l_1 = theta[1]
            l_2 = theta[2]
            l_3 = theta[3]
            l_4 = theta[4]
            l_5 = theta[5]
            l_6 = theta[6]

            log_A = (
                l_0
                + (l_1 * np.exp(l_2 * equivalence_ratio))
                + (
                    l_3
                    * np.tanh(
                        ((l_4 + (l_5 * equivalence_ratio)) * initial_temperature) + l_6
                    )
                )
            )

        else:
            raise ValueError("Number of parameters in the model not supported")

        return np.exp(log_A)

    def compute_Arrehenius_Ea(self, theta):
        """Function computes the Arrhenius activation energy"""
        if self.n_parameter_model == 4:
            Arrhenius_Ea = 30000 + 10000 * theta[3]
        elif self.n_parameter_model == 8:
            Arrhenius_Ea = 30000 + 10000 * theta[7]
        return Arrhenius_Ea

    def compute_model_prediction(self, theta, proc_log_file=None):
        """Function computes the model prediction, Temperature"""
        prediction = np.zeros((self.num_data_points, self.spatial_resolution))

        for idata_sample in range(self.num_data_points):

            initial_pressure = self.initial_pressure[idata_sample]
            initial_temperature = self.initial_temperature[idata_sample]
            equivalence_ratio = self.equivalence_ratio[idata_sample]

            # Compute the kinetic parameters
            kinetic_parameters = self.compute_kinetic_parameters(
                theta, initial_temperature, equivalence_ratio
            )

            if proc_log_file is not None:
                proc_log_file.write(
                    "     Arrhenius_A: {0:.18f}\n".format(kinetic_parameters["A"])
                )
                proc_log_file.flush()

            combustion_model = mech_2S_CH4_Westbrook(
                initial_temperature=initial_temperature,
                initial_pressure=initial_pressure,
                equivalence_ratio=equivalence_ratio,
                Arrhenius_A=kinetic_parameters["A"],
                Arrhenius_Ea=kinetic_parameters["Ea"],
            )

            ignition_time = combustion_model.compute_ignition_time()

            prediction[idata_sample, :] = np.log(ignition_time)

            if proc_log_file is not None:
                proc_log_file.write(
                    "     Prediction: {}\n".format(prediction[idata_sample, :])
                )
                proc_log_file.flush()

        return prediction

    def compute_temperature_prediction(self, theta_samples):
        """Function computes the model prediction, Temperature"""
        time_array = np.logspace(-6, 0, 3000)

        num_theta_samples = theta_samples.shape[1]

        prediction = np.zeros((3000, 4, num_theta_samples))
        for ii in range(num_theta_samples):
            print(
                "Computing prediction for theta sample: {0:d} / {1:d}".format(
                    ii, num_theta_samples
                ),
                flush=True,
            )
            for idata_sample in range(self.num_data_points):
                initial_pressure = self.initial_pressure[idata_sample]
                initial_temperature = self.initial_temperature[idata_sample]
                equivalence_ratio = self.equivalence_ratio[idata_sample]

                # Compute the Arrehnius rate
                Arrhenius_A = self.compute_Arrehenius_A(
                    theta=theta_samples[:, ii],
                    equivalence_ratio=equivalence_ratio,
                    initial_temperature=initial_temperature,
                )

                combustion_model = mech_2S_CH4_Westbrook(
                    initial_temperature=initial_temperature,
                    initial_pressure=initial_pressure,
                    equivalence_ratio=equivalence_ratio,
                    Arrhenius_A=Arrhenius_A,
                )

                try:
                    states, _ = combustion_model.mech_2S_CH4_Westbrook_combustion()
                    interpolated_temp = self.interpolate_state(
                        states.t, states.T, interpolate_time=time_array
                    )
                    prediction[:, idata_sample, ii] = interpolated_temp

                except:
                    prediction[:, idata_sample, ii] = np.nan * np.ones(3000)

        return prediction

    def compute_temperature_prediction_2S_Westbrook(self):
        """Function computes the model prediction, Temperature"""

        time_array = np.logspace(-6, 0, 3000)
        prediction = np.zeros((3000, 4))

        for idata_sample in range(self.num_data_points):
            initial_pressure = self.initial_pressure[idata_sample]
            initial_temperature = self.initial_temperature[idata_sample]
            equivalence_ratio = self.equivalence_ratio[idata_sample]

            combustion_model = mech_2S_CH4_Westbrook(
                initial_temperature=initial_temperature,
                initial_pressure=initial_pressure,
                equivalence_ratio=equivalence_ratio,
            )

            states, _ = combustion_model.mech_2S_CH4_Westbrook_combustion()
            interpolated_temp = self.interpolate_state(
                states.t, states.T, interpolate_time=time_array
            )
            prediction[:, idata_sample] = interpolated_temp

        return prediction

        """Function computes the true species evolution"""
        time_array = np.logspace(-6, 0, 3000)

        for idata_sample in range(self.num_data_points):
            initial_pressure = self.initial_pressure[idata_sample]
            initial_temperature = self.initial_temperature[idata_sample]
            equivalence_ratio = self.equivalence_ratio[idata_sample]

            combustion_model = mech_gri(
                initial_temperature=initial_temperature,
                initial_pressure=initial_pressure,
                equivalence_ratio=equivalence_ratio,
            )

            states, _ = combustion_model.gri_combustion()

            prediction[:, 0, ii] = self.interpolate_state(
                states.t, states("CH4").X.ravel(), interpolate_time=time_array
            )
            prediction[:, 1, ii] = self.interpolate_state(
                states.t, states("CO2").X.ravel(), interpolate_time=time_array
            )
            prediction[:, 2, ii] = self.interpolate_state(
                states.t, states("O2").X.ravel(), interpolate_time=time_array
            )
            prediction[:, 3, ii] = self.interpolate_state(
                states.t, states("CO").X.ravel(), interpolate_time=time_array
            )
            prediction[:, 4, ii] = self.interpolate_state(
                states.t, states("H2O").X.ravel(), interpolate_time=time_array
            )
            prediction[:, 5, ii] = self.interpolate_state(
                states.t, states("N2").X.ravel(), interpolate_time=time_array
            )

    def compute_species_evolution(
        self, theta_samples, initial_pressure, initial_temperature, equivalence_ratio
    ):
        """Function computes the model prediction, species concentration"""

        time_array = np.logspace(-6, 0, 3000)
        prediction_mean = np.zeros((3000, 6))
        prediction_std = np.zeros((3000, 6))
        prediction = np.zeros((3000, 6, theta_samples.shape[1]))

        for ii in range(theta_samples.shape[1]):

            # Compute the Arrehnius rate
            Arrhenius_A = self.compute_Arrehenius_A(
                theta=theta_samples[:, ii],
                equivalence_ratio=equivalence_ratio,
                initial_temperature=initial_temperature,
            )

            combustion_model = mech_2S_CH4_Westbrook(
                initial_temperature=initial_temperature,
                initial_pressure=initial_pressure,
                equivalence_ratio=equivalence_ratio,
                Arrhenius_A=Arrhenius_A,
            )

            states, _ = combustion_model.mech_2S_CH4_Westbrook_combustion()

            prediction[:, 0, ii] = self.interpolate_state(
                states.t, states("CH4").X.ravel(), interpolate_time=time_array
            )
            prediction[:, 1, ii] = self.interpolate_state(
                states.t, states("CO2").X.ravel(), interpolate_time=time_array
            )
            prediction[:, 2, ii] = self.interpolate_state(
                states.t, states("O2").X.ravel(), interpolate_time=time_array
            )
            prediction[:, 3, ii] = self.interpolate_state(
                states.t, states("CO").X.ravel(), interpolate_time=time_array
            )
            prediction[:, 4, ii] = self.interpolate_state(
                states.t, states("H2O").X.ravel(), interpolate_time=time_array
            )
            prediction[:, 5, ii] = self.interpolate_state(
                states.t, states("N2").X.ravel(), interpolate_time=time_array
            )

        prediction_mean = np.mean(prediction, axis=2)
        prediction_std = np.std(prediction, axis=2)

        return prediction_mean, prediction_std

    def compute_species_evolution_2S_Westbrook(self):
        """Function computes the true species evolution"""
        time_array = np.logspace(-6, 0, 3000)
        prediction = np.zeros((3000, 6, self.num_data_points))

        for idata in range(self.num_data_points):
            initial_pressure = self.initial_pressure[idata]
            initial_temperature = self.initial_temperature[idata]
            equivalence_ratio = self.equivalence_ratio[idata]

            combustion_model = mech_2S_CH4_Westbrook(
                initial_temperature=initial_temperature,
                initial_pressure=initial_pressure,
                equivalence_ratio=equivalence_ratio,
            )

            states, _ = combustion_model.mech_2S_CH4_Westbrook_combustion()

            prediction[:, 0, idata] = self.interpolate_state(
                states.t, states("CH4").X.ravel(), interpolate_time=time_array
            )
            prediction[:, 1, idata] = self.interpolate_state(
                states.t, states("CO2").X.ravel(), interpolate_time=time_array
            )
            prediction[:, 2, idata] = self.interpolate_state(
                states.t, states("O2").X.ravel(), interpolate_time=time_array
            )
            prediction[:, 3, idata] = self.interpolate_state(
                states.t, states("CO").X.ravel(), interpolate_time=time_array
            )
            prediction[:, 4, idata] = self.interpolate_state(
                states.t, states("H2O").X.ravel(), interpolate_time=time_array
            )
            prediction[:, 5, idata] = self.interpolate_state(
                states.t, states("N2").X.ravel(), interpolate_time=time_array
            )

        return prediction

    def compute_true_species_evolution(self):
        """Function computes the true species evolution"""
        time_array = np.logspace(-6, 0, 3000)
        prediction = np.zeros((3000, 6, self.num_data_points))

        for idata in range(self.num_data_points):
            initial_pressure = self.initial_pressure[idata]
            initial_temperature = self.initial_temperature[idata]
            equivalence_ratio = self.equivalence_ratio[idata]

            combustion_model = mech_gri(
                initial_temperature=initial_temperature,
                initial_pressure=initial_pressure,
                equivalence_ratio=equivalence_ratio,
            )

            states, _ = combustion_model.gri_combustion()

            prediction[:, 0, idata] = self.interpolate_state(
                states.t, states("CH4").X.ravel(), interpolate_time=time_array
            )
            prediction[:, 1, idata] = self.interpolate_state(
                states.t, states("CO2").X.ravel(), interpolate_time=time_array
            )
            prediction[:, 2, idata] = self.interpolate_state(
                states.t, states("O2").X.ravel(), interpolate_time=time_array
            )
            prediction[:, 3, idata] = self.interpolate_state(
                states.t, states("CO").X.ravel(), interpolate_time=time_array
            )
            prediction[:, 4, idata] = self.interpolate_state(
                states.t, states("H2O").X.ravel(), interpolate_time=time_array
            )
            prediction[:, 5, idata] = self.interpolate_state(
                states.t, states("N2").X.ravel(), interpolate_time=time_array
            )

        return prediction

    def compute_true_temperature_prediction(self):
        """Function computes the true Temperature"""

        time_array = np.logspace(-6, 0, 3000)
        prediction = np.zeros((3000, self.num_data_points))

        for idata_sample in range(self.num_data_points):
            print("True Temperature: ", idata_sample)
            initial_pressure = self.initial_pressure[idata_sample]
            initial_temperature = self.initial_temperature[idata_sample]
            equivalence_ratio = self.equivalence_ratio[idata_sample]

            combustion_model = mech_gri(
                initial_temperature=initial_temperature,
                initial_pressure=initial_pressure,
                equivalence_ratio=equivalence_ratio,
            )

            states, _ = combustion_model.gri_combustion()

            interpolated_temp = self.interpolate_state(
                states.t, states.T, interpolate_time=time_array
            )
            prediction[:, idata_sample] = interpolated_temp

        return prediction

    def interpolate_state(self, time, state, interpolate_time=None):
        """Function interpolates the temperature"""
        interpolated_state = np.interp(interpolate_time, time, state)
        return interpolated_state

    def compute_complete_state(self, theta):
        """Function computes the entire state"""
        state_list = []
        for idata_sample in range(self.num_data_points):
            Arrhenius_A = self.compute_Arrehenius_A(
                theta=theta,
                equivalence_ratio=self.equivalence_ratio[idata_sample],
                initial_temperature=self.initial_temperature[idata_sample],
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
        error_norm_sq = np.linalg.norm(error, axis=1) ** 2
        log_likelihood = (-0.5 / self.model_noise_cov) * (np.sum(error_norm_sq))
        return log_likelihood

    def compute_log_prior(self, theta):
        """Function computes the log prior"""
        error = (theta - self.prior_mean).reshape(-1, 1)
        error_norm_sq = (error.T @ np.linalg.solve(self.prior_cov, error)).item()
        log_prior = -0.5 * error_norm_sq
        return log_prior

    def compute_mle(self):
        """Function comptues the MLE"""
        theta_init = np.random.randn(self.num_parameters)

        def objective_function(theta):
            objective_function = -self.objective_scaling * self.compute_log_likelihood(
                theta
            )
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

        np.savetxt(
            save_prediction_path,
            prediction_save,
            delimiter=" ",
            header="Vartiables: Inital_temperature, Initial_pressure, Equivalence_ratio, Ignition_temperature",
        )
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
            objective_function = (
                -self.objective_scaling * self.compute_unnormalized_posterior(theta)
            )
            print("MAP objective : {0:.18e}".format(objective_function))
            return objective_function

        res = minimize(objective_function, theta_init, method="Nelder-Mead")
        res = minimize(objective_function, res.x)

        theta_map = res.x
        theta_map_cov = res.hess_inv

        return theta_map.reshape(-1, 1), theta_map_cov

    def compute_mcmc(self, theta_map, theta_map_cov, num_mcmc_samples):
        """Function computes the MCMC samples"""

        # num_mcmc_samples_per_proc =  int(num_mcmc_samles/size)

        compute_post = lambda theta: self.compute_unnormalized_posterior(theta)

        # mcmc_sampler = metropolis_hastings(
        #         initial_sample=theta_map.ravel(),
        #         target_log_pdf_evaluator=compute_post,
        #         num_samples=num_mcmc_samples_per_proc,
        #         # adapt_sample_threshold=500,
        #         initial_cov = 9e-3*theta_map_cov,
        #         )

        mcmc_sampler = adaptive_metropolis_hastings(
            initial_sample=theta_map.ravel(),
            target_log_pdf_evaluator=compute_post,
            num_samples=num_mcmc_samples,
            adapt_sample_threshold=10,
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

        # Plot the samples
        # fig, axs = plot_chains(burned_samples, title="Adaptive Metropolis Hastings")
        # plt.show()

        # Plot the histogram
        # labels = [r"$\theta_{0}$".format(i) for i in range(self.num_parameters)]
        # fig, axs, gs = scatter_matrix([burned_samples], labels=labels,
        #                         hist_plot=False, gamma=0.4)
        # fig.set_size_inches(15, 15)
        # plt.savefig(os.path.join(self.campaign_path, "mcmc_hist.png"), dpi=300)
        # plt.show()

        # burned_samples = sub_sample_data(mcmc_sampler.samples, frac_burn=0.6, frac_use=0.6)
        # comm.Barrier()

        # mcmc_samples = comm.gather(burned_samples, root=0)
        # if rank == 0:
        #     mcmc_samples = np.vstack(mcmc_samples)
        #     save_mcmc_samples_path = os.path.join(self.campaign_path, "mcmc_samples.npy")
        #     np.save(save_mcmc_samples_path, mcmc_samples)
        #     prediction = self.compute_temperature_prediction(theta_samples=mcmc_samples.T)
        #     save_mcmc_prediction_path = os.path.join(self.campaign_path, "mcmc_prediction.npy")
        #     np.save(save_mcmc_prediction_path, prediction)

    def plot_mcmc_estimate(self, mcmc_samples=None):
        time_array = np.logspace(-6, 0, 3000)

        # Plotting MCMC samples
        fig, axs = plt.subplots(self.num_parameters, 1, figsize=(15, 8))
        fig_save_path = os.path.join(self.campaign_path, "Figures/mcmc_samples.png")
        for i in range(self.num_parameters):
            axs[i].plot(mcmc_samples[:, i], color="k")
            axs[i].set_ylabel(r"$\Theta_{{{0:d}}}$".format(i + 1))
        plt.tight_layout()
        plt.savefig(fig_save_path)
        plt.close()

        # Compute model prediction for ALL MCMC samples
        # mcmc_prediction = self.compute_temperature_prediction(theta_samples=mcmc_samples.T)
        mcmc_prediction = np.load(
            os.path.join(self.campaign_path, "mcmc_prediction.npy")
        )

        # Compute feasibile samples
        feasible_idx = []
        for ii in range(self.num_data_points):
            sub_data = mcmc_prediction[:, ii, :]
            idx = np.arange(sub_data.shape[1])[
                np.logical_and(
                    np.max(sub_data, axis=0) < 1e4,
                    np.min(sub_data, axis=0) > self.initial_temperature[ii],
                )
            ]
            feasible_idx.append(idx)

        # Plot temperature prediction for feasible samples
        fig, axs = plt.subplots(1, 1, figsize=(15, 8))
        fig_save_path = os.path.join(self.campaign_path, "Figures/mcmc_prediction.png")

        true_temperature_prediction = self.compute_true_temperature_prediction()
        temperature_prediction_2S_Westbrook = (
            self.compute_temperature_prediction_2S_Westbrook()
        )

        for ii in range(self.num_data_points):

            sub_data = mcmc_prediction[:, ii, :]
            selected_data = sub_data[:, feasible_idx[ii]]
            mean = np.mean(selected_data, axis=1)
            std = np.std(selected_data, axis=1)
            ub = mean + std
            lb = mean - std
            axs.plot(
                time_array,
                true_temperature_prediction[:, ii],
                "-",
                color=dark_colors(ii),
                label="Gri-Mech 3.0, $T_{{o}}$={0:.1f} K".format(
                    self.initial_temperature[ii]
                ),
            )
            axs.plot(
                time_array,
                temperature_prediction_2S_Westbrook[:, ii],
                "-",
                dashes=[3, 2, 3, 2],
                lw=3,
                color=dark_colors(ii),
                label=r"2-step mechanism [Westbrook $\textit{{et al.}}$], $T_{{o}}$={0:.1f} K".format(
                    self.initial_temperature[ii]
                ),
            )
            axs.plot(
                time_array,
                mean,
                "-",
                dashes=[6, 2, 2, 2],
                color=dark_colors(ii),
                lw=3,
                label=r"$\mu_{{post}}$, $T_{{o}}$={0:.1f} K".format(
                    self.initial_temperature[ii]
                ),
            )
            axs.fill_between(
                np.logspace(-6, 0, 3000),
                ub,
                lb,
                ls="-",
                lw=2,
                color=dark_colors(ii),
                alpha=0.2,
            )

        axs.grid(True, axis="both", which="major", color="k", alpha=0.5)
        axs.grid(True, axis="both", which="minor", color="grey", alpha=0.3)
        axs.set_xscale("log")
        axs.set_ylim([1000, 4000])
        axs.set_xlim([1e-6, 1])
        axs.set_xlabel("Time [s]")
        axs.set_ylabel("Temperature [K]")
        plt.subplots_adjust(left=0.15, right=0.9, top=0.68, bottom=0.15)
        plt.legend(ncol=2, bbox_to_anchor=(0.525, 1.125, 0.5, 0.5))
        plt.savefig(fig_save_path)
        plt.close()

        # Plot species prediciton for feasible samples
        species_true_prediction = self.compute_true_species_evolution()
        species_2S_Westbrook_prediction = self.compute_species_evolution_2S_Westbrook()
        species_prediction_mean = np.zeros((3000, 6, self.num_data_points))
        species_prediction_std = np.zeros((3000, 6, self.num_data_points))

        for idata in range(self.num_data_points):
            initial_temperature = self.initial_temperature[idata]
            initial_pressure = self.initial_pressure[idata]
            equivalence_ratio = self.equivalence_ratio[idata]
            theta_samples = mcmc_samples[feasible_idx[idata], :].T

            species_mean, species_std = self.compute_species_evolution(
                theta_samples=theta_samples,
                initial_temperature=initial_temperature,
                initial_pressure=initial_pressure,
                equivalence_ratio=equivalence_ratio,
            )

            species_prediction_mean[:, :, idata] = species_mean
            species_prediction_std[:, :, idata] = species_std

        fig, axs = plt.subplots(2, 3, figsize=(19, 10))
        fig.tight_layout(h_pad=3.3, w_pad=3.3)
        fig_save_path = os.path.join(
            self.campaign_path, "Figures/mcmc_species_prediction.png"
        )

        for idata in range(self.num_data_points):

            axs[0, 0].plot(
                time_array,
                species_true_prediction[:, 0, idata],
                color=dark_colors(idata),
                label="Gri-Mech 3.0, $T_{{o}}$={0:.1f} K".format(
                    self.initial_temperature[idata]
                ),
            )
            axs[0, 0].plot(
                time_array,
                species_2S_Westbrook_prediction[:, 0, idata],
                "-",
                dashes=[2, 1, 2, 1],
                lw=3,
                color=dark_colors(idata),
                label=r"2-step mechanism [Westbrook $\textit{{et al.}}$], $T_{{o}}$={0:.1f} K".format(
                    self.initial_temperature[idata]
                ),
            )
            axs[0, 0].plot(
                time_array,
                species_prediction_mean[:, 0, idata],
                "-",
                dashes=[6, 2, 2, 2],
                color=dark_colors(idata),
                label=r"$\mu_{{post}}$, $T_{{o}}$={0:.1f} K".format(
                    self.initial_temperature[idata]
                ),
            )
            axs[0, 0].fill_between(
                time_array,
                species_prediction_mean[:, 0, idata]
                + species_prediction_std[:, 0, idata],
                species_prediction_mean[:, 0, idata]
                - species_prediction_std[:, 0, idata],
                ls="-",
                lw=2,
                color=dark_colors(idata),
                alpha=0.2,
            )
            axs[0, 0].set_xlabel(r"Time [s]")
            axs[0, 0].set_ylabel(r"[$CH_{4}$]")
            axs[0, 0].set_ylim([0, 0.1])
            axs[0, 0].set_xlim([1e-6, 1])
            axs[0, 0].set_xscale("log")
            axs[0, 0].yaxis.set_minor_locator(MultipleLocator(0.0125))
            # axs[0, 0].xaxis.set_minor_locator(MultipleLocator(0.5))
            # axs[0, 0].xaxis.set_minor_locator(AutoMinorLocator())
            axs[0, 0].grid(True, axis="both", which="major", color="k", alpha=0.5)
            axs[0, 0].grid(True, axis="both", which="minor", color="grey", alpha=0.3)

            axs[0, 1].plot(
                time_array,
                species_true_prediction[:, 1, idata],
                color=dark_colors(idata),
                label="Gri-Mech 3.0",
            )
            axs[0, 1].plot(
                time_array,
                species_2S_Westbrook_prediction[:, 1, idata],
                "-",
                dashes=[2, 1, 2, 1],
                lw=3,
                color=dark_colors(idata),
                label=r"$\mu_{{post}}$",
            )
            axs[0, 1].plot(
                time_array,
                species_prediction_mean[:, 1, idata],
                "-",
                dashes=[6, 2, 2, 2],
                color=dark_colors(idata),
                label=r"$\mu_{{post}}$",
            )
            axs[0, 1].fill_between(
                time_array,
                species_prediction_mean[:, 1, idata]
                + species_prediction_std[:, 1, idata],
                species_prediction_mean[:, 1, idata]
                - species_prediction_std[:, 1, idata],
                ls="-",
                lw=2,
                color=dark_colors(idata),
                alpha=0.2,
            )
            axs[0, 1].set_xlabel(r"Time [s]")
            axs[0, 1].set_ylabel(r"[$CO_{2}$]")
            axs[0, 1].set_ylim([0, 0.1])
            axs[0, 1].set_xlim([1e-6, 1])
            axs[0, 1].set_xscale("log")
            axs[0, 1].yaxis.set_minor_locator(MultipleLocator(0.0125))
            # axs[0, 1].xaxis.set_minor_locator(MultipleLocator(0.5))
            axs[0, 1].xaxis.set_minor_locator(AutoMinorLocator())
            axs[0, 1].grid(True, axis="both", which="major", color="k", alpha=0.5)
            axs[0, 1].grid(True, axis="both", which="minor", color="grey", alpha=0.3)

            axs[0, 2].plot(
                time_array,
                species_true_prediction[:, 2, idata],
                color=dark_colors(idata),
                label="Gri-Mech 3.0",
            )
            axs[0, 2].plot(
                time_array,
                species_2S_Westbrook_prediction[:, 2, idata],
                "-",
                dashes=[2, 1, 2, 1],
                lw=3,
                color=dark_colors(idata),
                label=r"$\mu_{{post}}$",
            )
            axs[0, 2].plot(
                time_array,
                species_prediction_mean[:, 2, idata],
                "-",
                dashes=[6, 2, 2, 2],
                color=dark_colors(idata),
                label=r"$\mu_{{post}}$",
            )
            axs[0, 2].fill_between(
                time_array,
                species_prediction_mean[:, 2, idata]
                + species_prediction_std[:, 2, idata],
                species_prediction_mean[:, 2, idata]
                - species_prediction_std[:, 2, idata],
                ls="-",
                lw=2,
                color=dark_colors(idata),
                alpha=0.2,
            )
            axs[0, 2].set_xlabel(r"Time [s]")
            axs[0, 2].set_ylabel(r"[$O_{2}$]")
            axs[0, 2].set_ylim([0, 0.2])
            axs[0, 2].set_xlim([1e-6, 1])
            axs[0, 2].set_xscale("log")
            axs[0, 2].yaxis.set_minor_locator(MultipleLocator(0.025))
            # axs[0, 2].xaxis.set_minor_locator(MultipleLocator(0.5))
            axs[0, 2].xaxis.set_minor_locator(AutoMinorLocator())
            axs[0, 2].grid(True, axis="both", which="major", color="k", alpha=0.5)
            axs[0, 2].grid(True, axis="both", which="minor", color="grey", alpha=0.3)

            axs[1, 0].plot(
                time_array,
                species_true_prediction[:, 3, idata],
                color=dark_colors(idata),
                label="Gri-Mech 3.0",
            )
            axs[1, 0].plot(
                time_array,
                species_2S_Westbrook_prediction[:, 3, idata],
                "-",
                dashes=[2, 1, 2, 1],
                lw=3,
                color=dark_colors(idata),
                label=r"$\mu_{{post}}$",
            )
            axs[1, 0].plot(
                time_array,
                species_prediction_mean[:, 3, idata],
                "-",
                dashes=[6, 2, 2, 2],
                color=dark_colors(idata),
                label=r"$\mu_{{post}}$",
            )
            axs[1, 0].fill_between(
                time_array,
                species_prediction_mean[:, 3, idata]
                + species_prediction_std[:, 3, idata],
                species_prediction_mean[:, 3, idata]
                - species_prediction_std[:, 3, idata],
                ls="-",
                lw=2,
                color=dark_colors(idata),
                alpha=0.2,
            )
            axs[1, 0].set_xlabel(r"Time [s]")
            axs[1, 0].set_ylabel(r"[$CO$]")
            axs[1, 0].set_ylim([0, 0.2])
            axs[1, 0].set_xlim([1e-6, 1])
            axs[1, 0].set_xscale("log")
            axs[1, 0].yaxis.set_minor_locator(MultipleLocator(0.02))
            axs[1, 0].yaxis.set_major_locator(MultipleLocator(0.04))
            axs[1, 0].xaxis.set_minor_locator(AutoMinorLocator())
            # axs[1, 0].xaxis.set_minor_locator(MultipleLocator(0.5))
            axs[1, 0].grid(True, axis="both", which="major", color="k", alpha=0.5)
            axs[1, 0].grid(True, axis="both", which="minor", color="grey", alpha=0.3)

            axs[1, 1].plot(
                time_array,
                species_true_prediction[:, 4, idata],
                color=dark_colors(idata),
                label="Gri-Mech 3.0",
            )
            axs[1, 1].plot(
                time_array,
                species_2S_Westbrook_prediction[:, 4, idata],
                "-",
                dashes=[2, 1, 2, 1],
                lw=3,
                color=dark_colors(idata),
                label=r"$\mu_{{post}}$",
            )
            axs[1, 1].plot(
                time_array,
                species_prediction_mean[:, 4, idata],
                "-",
                dashes=[6, 2, 2, 2],
                color=dark_colors(idata),
                label=r"$\mu_{{post}}$",
            )
            axs[1, 1].fill_between(
                time_array,
                species_prediction_mean[:, 4, idata]
                + species_prediction_std[:, 4, idata],
                species_prediction_mean[:, 4, idata]
                - species_prediction_std[:, 4, idata],
                ls="-",
                lw=2,
                color=dark_colors(idata),
                alpha=0.2,
            )
            axs[1, 1].set_xlabel(r"Time [s]")
            axs[1, 1].set_ylabel(r"[$H_{2}O$]")
            axs[1, 1].set_ylim([0, 0.2])
            axs[1, 1].set_xlim([1e-6, 1])
            axs[1, 1].set_xscale("log")
            axs[1, 1].yaxis.set_minor_locator(MultipleLocator(0.025))
            # axs[1, 1].yaxis.set_major_locator(MultipleLocator(0.04))
            axs[1, 1].xaxis.set_minor_locator(AutoMinorLocator())
            # axs[1, 1].xaxis.set_minor_locator(MultipleLocator(0.5))
            axs[1, 1].grid(True, axis="both", which="major", color="k", alpha=0.5)
            axs[1, 1].grid(True, axis="both", which="minor", color="grey", alpha=0.3)

            axs[1, 2].plot(
                time_array,
                species_true_prediction[:, 5, idata],
                color=dark_colors(idata),
                label="Gri-Mech 3.0",
            )
            axs[1, 2].plot(
                time_array,
                species_2S_Westbrook_prediction[:, 5, idata],
                "-",
                dashes=[2, 1, 2, 1],
                lw=3,
                color=dark_colors(idata),
                label=r"$\mu_{{post}}$",
            )
            axs[1, 2].plot(
                time_array,
                species_prediction_mean[:, 5, idata],
                "-",
                dashes=[6, 2, 2, 2],
                color=dark_colors(idata),
                label=r"$\mu_{{post}}$",
            )
            axs[1, 2].fill_between(
                time_array,
                species_prediction_mean[:, 5, idata]
                + species_prediction_std[:, 5, idata],
                species_prediction_mean[:, 5, idata]
                - species_prediction_std[:, 5, idata],
                ls="-",
                lw=2,
                color=dark_colors(idata),
                alpha=0.2,
            )
            axs[1, 2].set_xlabel(r"Time [s]")
            axs[1, 2].set_ylabel(r"[$N_{2}$]")
            axs[1, 2].set_ylim([0.65, 0.75])
            axs[1, 2].set_xlim([1e-6, 1])
            axs[1, 2].set_xscale("log")
            axs[1, 2].yaxis.set_minor_locator(MultipleLocator(0.025))
            # axs[12 1].yaxis.set_major_locator(MultipleLocator(0.04))
            axs[1, 2].xaxis.set_minor_locator(AutoMinorLocator())
            # axs[1, 2].xaxis.set_minor_locator(MultipleLocator(0.5))
            axs[1, 2].grid(True, axis="both", which="major", color="k", alpha=0.5)
            axs[1, 2].grid(True, axis="both", which="minor", color="grey", alpha=0.3)

        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.35, 0.5, 0.5, 0.5), ncol=2)
        plt.subplots_adjust(top=0.73, bottom=0.12, left=0.1, right=0.95)
        plt.savefig(fig_save_path)
        plt.close()

    def plot_map_estimate(self, theta_map, theta_map_cov):
        """Function plots the map estimate"""
        num_samples = 50
        temperature_range = np.linspace(833, 2500, num_samples)
        calibrated_model_prediction = np.zeros(
            (num_samples, temperature_range.shape[0])
        )
        un_calibrated_model_prediction = np.zeros_like(temperature_range)
        gri_prediction = np.zeros_like(temperature_range)

        d = theta_map.shape[0]
        std_normal_samples = np.random.randn(d, num_samples)
        cov_cholesky = np.linalg.cholesky(theta_map_cov)
        theta_samples = theta_map + cov_cholesky @ std_normal_samples

        for ii in range(num_samples):
            print("sample : {}".format(ii))
            eval_theta = theta_samples[:, ii]
            for jj, itemp in enumerate(temperature_range):

                Arrhenius_A = self.compute_Arrehenius_A(
                    theta=eval_theta, equivalence_ratio=1.0, initial_temperature=itemp
                )

                model_2S_CH4_Westbrook = mech_2S_CH4_Westbrook(
                    initial_temperature=itemp,
                    initial_pressure=100000,
                    equivalence_ratio=1.0,
                    Arrhenius_A=Arrhenius_A,
                )

                calibrated_model_prediction[ii, jj] = np.log(
                    model_2S_CH4_Westbrook.compute_ignition_time(internal_state="plot")
                )

        for ii, itemp in enumerate(temperature_range):
            # Compute gri prediciton
            gri_model = mech_gri(
                initial_temperature=itemp,
                initial_pressure=100000,
                equivalence_ratio=1.0,
            )
            gri_prediction[ii] = np.log(gri_model.compute_ignition_time())

            # Compute 2-step Westbrook prediction (uncalibrated)
            model_2S_CH4_Westbrook = mech_2S_CH4_Westbrook(
                initial_temperature=itemp,
                initial_pressure=100000,
                equivalence_ratio=1.0,
            )

            un_calibrated_model_prediction[ii] = np.log(
                model_2S_CH4_Westbrook.compute_ignition_time(internal_state="plot")
            )

        mean_val = np.mean(calibrated_model_prediction, axis=0)
        std_val = np.std(calibrated_model_prediction, axis=0)

        save_fig_path = os.path.join(
            self.campaign_path,
            "Figures/prediction_map_phi_{}_pressure_{}_noise_{}.png".format(
                1.0, 100000, self.model_noise_cov
            ),
        )
        fig, axs = plt.subplots(figsize=(11, 7))
        axs.scatter(
            1000 / self.initial_temperature,
            self.ytrain.ravel(),
            label="Data",
            color="k",
            marker="s",
            s=100,
        )
        axs.plot(
            1000 / temperature_range,
            gri_prediction,
            "--",
            label="Gri-Mech 3.0",
            color="k",
        )
        axs.plot(
            1000 / temperature_range,
            un_calibrated_model_prediction,
            label=r"2-step mechanism [Westbrook \textit{et al.}]",
            color="r",
        )
        axs.plot(1000 / temperature_range, mean_val, label=r"$\mu_{MAP}$", color="b")
        axs.fill_between(
            1000 / temperature_range,
            mean_val + std_val,
            mean_val - std_val,
            ls="--",
            edgecolor="C0",
            lw=2,
            alpha=0.3,
            label=r"$\pm\sigma_{MAP}$",
        )
        axs.set_xlabel(r"1000/T [1/K]")
        axs.set_ylabel(r"$\log(t_{ign})$")
        axs.legend(loc="upper left")
        axs.grid(True, axis="both", which="major", color="k", alpha=0.5)
        axs.grid(True, axis="both", which="minor", color="grey", alpha=0.3)
        axs.xaxis.set_minor_locator(MultipleLocator(0.05))
        axs.xaxis.set_major_locator(MultipleLocator(0.2))
        axs.yaxis.set_minor_locator(AutoMinorLocator())
        axs.set_xlim([0.4, 1.2])
        axs.set_ylim([-15, 10])
        plt.tight_layout()
        plt.savefig(save_fig_path)
        plt.close()

    def plot_mle_estimate(self, theta_mle):
        """Function plots the mle estimate"""
        num_samples = 50
        temperature_range = np.linspace(833, 2500, num_samples)
        calibrated_model_prediction = np.zeros_like(temperature_range)
        un_calibrated_model_prediction = np.zeros_like(temperature_range)
        gri_prediction = np.zeros_like(temperature_range)

        for ii, itemp in enumerate(temperature_range):
            # Compute gri prediciton
            gri_model = mech_gri(
                initial_temperature=itemp,
                initial_pressure=100000,
                equivalence_ratio=1.0,
            )
            gri_prediction[ii] = np.log(gri_model.compute_ignition_time())

            # Compute 2-step Westbrook prediction (Calibrated)
            Arrhenius_A = self.compute_Arrehenius_A(
                theta=theta_mle, equivalence_ratio=1.0, initial_temperature=itemp
            )

            model_2S_CH4_Westbrook = mech_2S_CH4_Westbrook(
                initial_temperature=itemp,
                initial_pressure=100000,
                equivalence_ratio=1.0,
                Arrhenius_A=Arrhenius_A,
            )

            calibrated_model_prediction[ii] = np.log(
                model_2S_CH4_Westbrook.compute_ignition_time(internal_state="plot")
            )

            # Compute 2-step Westbrook prediction (uncalibrated)
            model_2S_CH4_Westbrook = mech_2S_CH4_Westbrook(
                initial_temperature=itemp,
                initial_pressure=100000,
                equivalence_ratio=1.0,
            )

            un_calibrated_model_prediction[ii] = np.log(
                model_2S_CH4_Westbrook.compute_ignition_time(internal_state="plot")
            )

        save_fig_path = os.path.join(
            self.campaign_path,
            "Figures/prediction_mle_phi_{}_pressure_{}_noise_{}.png".format(
                1.0, 100000, self.model_noise_cov
            ),
        )
        fig, axs = plt.subplots(figsize=(12, 7))
        axs.scatter(
            1000 / self.initial_temperature,
            self.ytrain.ravel(),
            label="Data",
            color="k",
            marker="s",
            s=100,
        )
        axs.plot(
            1000 / temperature_range,
            gri_prediction,
            "--",
            label="Gri-Mech 3.0",
            color="k",
        )
        axs.plot(
            1000 / temperature_range,
            calibrated_model_prediction,
            label="M.L.E",
            color="b",
        )
        axs.plot(
            1000 / temperature_range,
            un_calibrated_model_prediction,
            label=r"2-step mechanism [Westbrook \textit{et al.}]",
            color="r",
        )
        axs.set_xlabel(r"1000/T [1/K]")
        axs.set_ylabel(r"$\log(t_{ign})$")
        axs.legend(loc="upper left")
        axs.grid(True, axis="both", which="major", color="k", alpha=0.5)
        axs.grid(True, axis="both", which="minor", color="grey", alpha=0.3)
        axs.set_xlim([0.4, 1.2])
        axs.set_ylim([-15, 10])
        axs.xaxis.set_minor_locator(MultipleLocator(0.05))
        axs.xaxis.set_major_locator(MultipleLocator(0.2))
        axs.yaxis.set_minor_locator(AutoMinorLocator())
        plt.tight_layout()
        plt.savefig(save_fig_path)
        plt.close()

    def compute_model_identifiability(self, prior_mean, prior_cov):
        """Function computes the model identifiability"""

        def forward_model(theta, proc_log_file=None):
            prediction = self.compute_model_prediction(
                theta, proc_log_file=proc_log_file
            ).T
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

        # estimator.compute_individual_parameter_data_mutual_information_via_mc(
        #         use_quadrature=True,
        #         single_integral_gaussian_quad_pts=5
        #         )

        estimator.compute_posterior_pair_parameter_mutual_information(
            use_quadrature=True,
            single_integral_gaussian_quad_pts=5,
            double_integral_gaussian_quad_pts=10,
        )


def load_configuration_file(config_file_path="./config.yaml"):
    """Function loads the configuration file"""
    with open(config_file_path, "r") as config_file:
        config_data = yaml.safe_load(config_file)
    if rank == 0:
        print("Loaded %s configuration file" % (config_file_path), flush=True)
    return config_data


def main():
    # Load the configurations
    config_data = load_configuration_file()

    # Model prior
    num_model_parameters = config_data["n_parameter_model"]
    prior_mean = np.zeros(num_model_parameters)
    prior_cov = np.eye(num_model_parameters)

    # Campaign path
    campaign_path = os.path.join(
        os.getcwd(), "campaign_results/campaign_%d" % (config_data["campaign_id"])
    )

    # Learning model
    learning_model = learn_ignition_model(
        config_data=config_data,
        campaign_path=campaign_path,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
    )

    if config_data["compute_mle"]:
        theta_mle = learning_model.compute_mle()
        np.save(os.path.join(campaign_path, "theta_mle.npy"), theta_mle)
    elif config_data["compute_map"]:
        theta_map, theta_map_cov = learning_model.compute_map()
        np.save(os.path.join(campaign_path, "theta_map.npy"), theta_map)
        np.save(os.path.join(campaign_path, "theta_map_cov.npy"), theta_map_cov)
    elif config_data["compute_mcmc"]:
        num_mcmc_samples = config_data["num_mcmc_samples"]
        theta_map = np.load(os.path.join(campaign_path, "theta_map.npy"))
        theta_map_cov = np.load(os.path.join(campaign_path, "theta_map_cov.npy"))
        learning_model.compute_mcmc(
            theta_map, theta_map_cov, num_mcmc_samples=num_mcmc_samples
        )

    if "--plotmap" in sys.argv:
        theta_map = np.load(os.path.join(campaign_path, "theta_map.npy"))
        theta_map_cov = np.load(os.path.join(campaign_path, "theta_map_cov.npy"))
        learning_model.plot_map_estimate(
            theta_map=theta_map, theta_map_cov=theta_map_cov
        )

    elif "--plotmle" in sys.argv:
        theta_mle = np.load(os.path.join(campaign_path, "theta_mle.npy"))
        learning_model.plot_mle_estimate(
            theta_mle=theta_mle,
        )

    elif "--plotmcmc" in sys.argv:
        mcmc_samples = np.load(os.path.join(campaign_path, "mcmc_samples.npy"))
        learning_model.plot_mcmc_estimate(
            mcmc_samples=mcmc_samples,
        )

    if config_data["compute_identifiability"]:
        theta_map = np.load(os.path.join(campaign_path, "theta_map.npy"))
        theta_map_cov = np.load(os.path.join(campaign_path, "theta_map_cov.npy"))
        learning_model.compute_model_identifiability(
            prior_mean=theta_map, prior_cov=theta_map_cov
        )

    if rank == 0:
        learning_model.log_file.close()


if __name__ == "__main__":
    main()
