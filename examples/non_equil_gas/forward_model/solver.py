import torch
import torch.nn as nn
import numpy as np
import os.path as osp
import yaml
import warnings


class SOLVER(nn.Module):
    boltzmann_constant = 1.38064852e-23  # J/K #IUKB
    boltzmann_constant_ev = 8.617333262145e-5  # eV/K #K
    planck_constant = 6.62607015e-34  # J s #IUH
    avagaadro_constant = 6.02214076e23  # mol^-1 #IUNA
    pi = np.pi  # IUPI
    universal_gas_constant = 8.3144598  # J/(mol K) #URG or IUR
    electron_charge = 1.602191e-19  # C #IUE

    def __init__(self, config_data, device):
        self.system = config_data["SYSTEM"]
        self.ambient_temp = config_data["AMBIENT_TEMP"]
        self.ambient_pressure = config_data["AMBIENT_PRESSURE"]
        self.initial_temp = config_data["INITIAL_TEMP"]
        self.num_species = config_data["NUM_SPECIES"]

        self.num_bins = config_data["NUM_BINS"]

        self.system_path = osp.join(
            config_data["BASE_PATH"], "system_definition/" + self.system + "_system"
        )
        system_parameter_file_path = osp.join(self.system_path, "system_parameters.yml")

        # Get system-specific parameters
        print("Evaluating {} system".format(self.system))
        if osp.exists(system_parameter_file_path):
            with open(system_parameter_file_path, "r") as config_file:
                self.system_config_data = yaml.load(config_file, Loader=yaml.FullLoader)
        else:
            raise ValueError("System parameter file does not exist")

        # GAT
        self.ground_state_degeneracy = self.system_config_data[
            "GROUND_STATE_DEGENERACY"
        ]
        # EAT
        self.activation_energy = self.system_config_data["ACTIVATION_ENERGY"]

        self.num_states = self.system_config_data["NUM_STATES"]

        if self.system == "H2_H":
            self.species_list = ["H2", "H"]
            self.species_molar_mass = [
                self.system_config_data["MOLAR_MASS_H2"],
                self.system_config_data["MOLAR_MASS_H"],
            ]
            self.molecular_mass = self.system_config_data["MOLECULAR_MASS_H"]
        elif self.system == "N2_N":
            raise NotImplementedError
            self.species_list = ["N2", "N"]
            self.species_molar_mass = [
                self.system_config_data["MOLAR_MASS_N2"],
                self.system_config_data["MOLAR_MASS_N"],
            ]
            self.molecular_mass = self.system_config_data["MOLECULAR_MASS_N"]
        else:
            raise Exception("System {} not implemented".format(self.system))

        self.specific_gas_constant = [
            self.universal_gas_constant / self.species_molar_mass[i]
            for i in range(self.num_species)
        ]

        # State to state solution
        self.state_to_state_sol = self._load_state_to_state_sol()

        # Energy states
        self.energy_states, self.degeneracy = self._load_energy_states()

        # Dissociation coefficients
        dissociation_arr_params = self._load_dissociation_arr_params()
        self.dissociation_rate = self._comp_arrhenius_rate(
            arrhenius_params=dissociation_arr_params, temperature=self.ambient_temp
        )

        self.dissociation_rate = self.dissociation_rate * 1e-6  # Convert cm3 to m3

        # Compute the recombination rate from the dissociation rate
        self.recombination_rate = self._comp_recombination_rate()

        # Excitation-Deexcitation rates coefficients
        (
            self.excitation_deexitation_index,
            excitation_arr_params,
        ) = self._load_excitation_deexcitation_arr_params()

        self.excitation_rate = self._comp_arrhenius_rate(
            arrhenius_params=excitation_arr_params, temperature=self.ambient_temp
        )

        self.excitation_rate = self.excitation_rate * 1e-6  # Convert cm3 to m3

        self.deexcitation_rate = self._comp_deexcitation_rate()

    def _comp_recombination_rate(self):
        """Compute the recombination rate from the dissociation rate"""

        assert (
            self.num_species == 2
        ), "Recombination rate only implemented for 2 species"

        fac1 = (
            2
            * self.pi
            * self.universal_gas_constant
            / (self.planck_constant * self.avagaadro_constant) ** 2
        )
        warnings.warn("QMOL and QAT are the same thing!!! (Check this)")
        QMOL = (fac1 * self.species_molar_mass[0] * self.ambient_temp) ** (1.5)
        QAT = (fac1 * self.species_molar_mass[0] * self.ambient_temp) ** (1.5)

        term1 = (self.ground_state_degeneracy * QAT) ** 2
        term2 = torch.exp(
            -(2 * self.activation_energy / self.avagaadro_constant - self.energy_states)
            / (self.boltzmann_constant * self.ambient_temp)
        )
        term3 = self.degeneracy * QMOL
        Keq = term1 * term2 / term3
        recom_rate = self.dissociation_rate / Keq

        assert (
            recom_rate.shape[0] == self.num_states
        ), "Recombination rate shape does not match energy states"
        assert recom_rate.shape[1] == 1, "Recombination rate must be a column vector"

        return recom_rate

    def _load_state_to_state_sol(self):
        """Function loads the state to state solution"""
        path = osp.join(self.system_path, "state_to_state_solution.dat")
        data = torch.from_numpy(np.loadtxt(path, skiprows=10))[
            :, 1 : self.num_species + 1
        ]
        return data

    def _load_energy_states(self):
        """Function loads the energy states"""
        path = osp.join(self.system_path, "energy_states.dat")
        data = np.loadtxt(path, skiprows=2)
        energy_states = torch.from_numpy(data[:, 1] * self.electron_charge).view(-1, 1)
        degeneracy = torch.from_numpy(data[:, 2]).view(-1, 1)

        assert (
            energy_states.shape[0] == self.num_states
        ), "Number of energy states does not match"
        assert (
            degeneracy.shape[0] == self.num_states
        ), "Number of degeneracy states does not match"

        assert energy_states.shape[1] == 1, "Energy states should be a column vector"
        assert degeneracy.shape[1] == 1, "degeneracy should be a column vector"

        return energy_states, degeneracy

    def _load_dissociation_arr_params(self):
        """Function loads the dissociation arrehenius parameters"""
        path = osp.join(self.system_path, "Kinetics_Data/dissociation_rate_arr_fit.dat")
        data = np.loadtxt(path, skiprows=3)[:, 1:]

        assert data.shape[0] == self.num_states, "Number of states does not match"
        assert data.shape[1] == 3, "Provide 3 parameters for the arrhenius fit"

        return torch.from_numpy(data)

    def _load_excitation_deexcitation_arr_params(self):
        """Function loads the excitation arrehenius parameters"""
        path = osp.join(self.system_path, "Kinetics_Data/excitation_rate_arr_fit.dat")
        data = np.loadtxt(path, skiprows=3)
        excitation_deexitation_index = data[:, :2].astype(int)
        excitation_arr_params = data[:, 2:]

        return excitation_deexitation_index, torch.from_numpy(excitation_arr_params)

    def _comp_arrhenius_rate(self, arrhenius_params, temperature):
        """Function computes the Arrhenius rate from the Arrhenius parameters"""
        rate = arrhenius_params[:, 0] * torch.exp(
            arrhenius_params[:, 1] * np.log(temperature)
            - arrhenius_params[:, 2] / temperature
        )
        rate = rate.view(-1, 1)

        # assert(rate.shape[0] == self.num_states), 'Number of states does not match'
        assert rate.shape[1] == 1, "Rate should be a column vector"

        return rate
