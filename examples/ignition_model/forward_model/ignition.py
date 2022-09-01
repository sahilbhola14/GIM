# Constant volume solve
import numpy as np
import matplotlib.pyplot as plt
import yaml


class ignition():
    def __init__(
            self,
            pressure,
            temp_init,
            num_species,
            species_mole_fraction_init,
            species_molar_mass,
            nasa_polynomials,
            equivalence_ratio,
            pre_exponential_factor_parameters,
            species_reference_enthalpy
    ):
        # Global
        self.pressure = pressure
        self.temp_init = temp_init

        # Species
        self.num_species = num_species
        self.species_mole_fraction_init = species_mole_fraction_init
        self.species_molar_mass = species_molar_mass
        self.nasa_polynomials = nasa_polynomials
        self.species_reference_enthalpy = species_reference_enthalpy

        # Constants
        self.ideal_gas_constant = 8.314472  # m^2 kgs^-2 K^-1 mol^-1
        self.equivalence_ratio = equivalence_ratio
        self.pre_exponential_factor_parameters = pre_exponential_factor_parameters

        assert(len(self.species_mole_fraction_init.keys()) ==
               self.num_species), "Provide initial mole fraction for each species"

        h = self.compute_mixture_internal_energy(
                concentration=self.species_mole_fraction_init,
                temperature=self.temp_init
                )
        print(h*4.2*1000)

    def compute_total_moles(self, concentration):
        """Function comptues the total number of moles"""
        total_moles = 0
        for ispecies in concentration.keys():
            total_moles += concentration[str(ispecies)]
        return total_moles

    def compute_mole_fraction(self, concentration):
        """Function computes the mole fraction (assuming unit vol.)"""
        mole_fraction = np.array([concentration[str(ispecies)]
                                 for ispecies in concentration.keys()])
        total_moles = self.compute_total_moles(concentration=concentration)
        return mole_fraction / total_moles

    def compute_mass_fraction(self, concentration):
        """Function comptues the mass fraction"""
        mole_fraction = self.compute_mole_fraction(concentration=concentration)
        mixture_molar_mass = self.compute_mixture_molar_mass(
            concentration=concentration)
        mass_fraction = np.array([mole_fraction[ii]*self.species_molar_mass[str(ispecies)]
                                 for ii, ispecies in enumerate(concentration.keys())])
        mass_fraction = mass_fraction / mixture_molar_mass
        return mass_fraction

    def compute_mixture_molar_mass(self, concentration):
        """Function computes the mixture molar mass"""
        mixture_molar_mass = 0
        mole_fraction = self.compute_mole_fraction(concentration)

        for ii, ispecies in enumerate(concentration.keys()):
            mixture_molar_mass += mole_fraction[ii] * \
                self.species_molar_mass[str(ispecies)]

        return mixture_molar_mass

    def weight_with_mass_fraction(self, concentration, quantity):
        """Function weights the quantity with mass fraction"""
        mass_fraction = self.compute_mass_fraction(concentration=concentration)
        weighted_sum = np.sum(mass_fraction*quantity)
        return weighted_sum

    def weight_with_mole_fraction(self, concentration, quantity):
        """Function weights the quantity with mole fraction"""
        mole_fraction = self.compute_mole_fraction(concentration=concentration)
        weighted_sum = np.sum(mole_fraction*quantity)
        return weighted_sum

    def get_nasa_polynomials(self, species_name, temperature):
        """Function returns the nasa polynomials"""
        available_list = list(self.nasa_polynomials.keys())
        species_found = False
        temperature_range_found = False
        nasa_polynomials = None
        for ispecies in available_list:
            if ispecies == species_name:
                species = self.nasa_polynomials[ispecies]
                num_temp_ranges = species['num_temp_ranges']
                avail_temp_ranges = [species['nasa_coeffs'][ii]
                                     ['temp_range'] for ii in range(num_temp_ranges)]

                for itemp_range in range(num_temp_ranges):
                    temperature_range = avail_temp_ranges[itemp_range]
                    if itemp_range == num_temp_ranges-1:
                        check = np.logical_and(
                            temperature >= temperature_range[0], temperature <= temperature_range[1])
                        if check:
                            temperature_range_found = True
                    else:
                        check = np.logical_and(
                            temperature >= temperature_range[0], temperature < temperature_range[1])
                        if check:
                            temperature_range_found = True
                    if temperature_range_found:
                        nasa_polynomials = species['nasa_coeffs'][itemp_range]['coefficients']
                        break
                species_found = True
                break
        if species_found is False:
            raise ValueError("Nasa polynomials for the species not found")
        if temperature_range_found is False:
            raise ValueError("Temperature out of range")

        return nasa_polynomials

    def compute_species_specific_heat_at_constant_pressure(self, species_name, temperature=None, temperature_stale=None):
        """Function computes the specific heat at constant pressure
        Units: cal/mol*K"""
        def compute_cp(nasa_polynomials, temperature):
            a1 = nasa_polynomials[0]
            a2 = nasa_polynomials[1]
            a3 = nasa_polynomials[2]
            a4 = nasa_polynomials[3]
            a5 = nasa_polynomials[4]
            species_cp = a1 + a2*temperature + a3*temperature**2 + \
                a4*temperature**3 + a5*temperature**4
            species_cp = species_cp*self.ideal_gas_constant
            return species_cp

        if temperature is not None:
            # Point estimate
            nasa_polynomials = self.get_nasa_polynomials(
                species_name=species_name, temperature=temperature)
            species_cp = compute_cp(
                nasa_polynomials=nasa_polynomials, temperature=temperature)  # Units: J/mol*K
            species_cp = species_cp / 4.2  # Units: cal/mol*K
        else:
            # For preparing a function
            nasa_polynomials = self.get_nasa_polynomials(
                species_name=species_name, temperature=temperature_stale)
            raise ValueError("Under construction")

        return species_cp

    def compute_species_specific_heat_at_constant_volume(self, species_name, temperature=None, temperature_stale=None):
        """Function computes the specific heat at constant volume
        Units: cal/mol*K"""
        ideal_gas_constant_cal = self.ideal_gas_constant / \
            4.2  # Ideal gas constant in cal/K*mol
        species_cp = self.compute_species_specific_heat_at_constant_pressure(
            species_name=species_name,
            temperature=temperature,
            temperature_stale=temperature_stale
        )
        species_cv = species_cp - ideal_gas_constant_cal
        return species_cv

    def compute_species_enthalpy(self, species_name, temperature=None, temperature_stale=None):
        """Function computes the enthalpy of the species using NASA polynomials (cal/mol)"""
        def compute_enthalpy(nasa_polynomials, temperature):
            a1 = nasa_polynomials[0]
            a2 = nasa_polynomials[1]
            a3 = nasa_polynomials[2]
            a4 = nasa_polynomials[3]
            a5 = nasa_polynomials[4]
            a6 = nasa_polynomials[5]
            species_enthalpy = a1 + (0.5*a2*temperature) + \
                (a3/3)*(temperature**2) + (a4/4)*(temperature**3) + \
                (a5/5)*temperature**4 + (a6/temperature)
            species_enthalpy = species_enthalpy*temperature*self.ideal_gas_constant
            return species_enthalpy

        if temperature is not None:
            nasa_polynomials = self.get_nasa_polynomials(
                species_name=species_name, temperature=temperature)
            species_enthalpy = compute_enthalpy(
                nasa_polynomials=nasa_polynomials, temperature=temperature)  # Units: J/mol
            species_enthalpy = species_enthalpy/4.2  # Units: cal/mol

        return species_enthalpy

    def compute_species_internal_energy(self, species_name, temperature=None, temperature_stale=None):
        """Function computes the internal energy of the species (cal/mol)"""
        ideal_gas_constant_cal = self.ideal_gas_constant/4.2
        species_enthalpy = self.compute_species_enthalpy(
                species_name=species_name,
                temperature=temperature,
                temperature_stale=temperature_stale
                )
        species_internal_energy = species_enthalpy - ideal_gas_constant_cal*temperature
        return species_internal_energy

    def compute_mixture_specific_heat_at_constant_pressure(self, concentration, temperature=None, temperature_stale=None):
        """Function computes the mixture specific heat at constant pressure (cal/mol*K)"""
        # Compute species specific heat at constant pressure, cp (cal/mol*K)
        species_cp = np.zeros(self.num_species)
        for ii, ispecies in enumerate(concentration.keys()):
            species_cp[ii] = self.compute_species_specific_heat_at_constant_pressure(
                species_name=ispecies,
                temperature=self.temp_init
            )
        mixture_cp = self.weight_with_mole_fraction(
            concentration=concentration, quantity=species_cp)  # cal/mol*K

        return mixture_cp

    def compute_mixture_specific_heat_at_constant_volume(self, concentration, temperature=None, temperature_stale=None):
        """Function comptues the mixture specific heat at constant volume (cal/mol*K)"""
        mixture_cp = self.compute_mixture_specific_heat_at_constant_pressure(
            concentration=concentration,
            temperature=temperature
        )
        ideal_gas_constant_cal = self.ideal_gas_constant/4.2
        mixture_cv = mixture_cp - ideal_gas_constant_cal
        return mixture_cv

    def compute_mixture_enthalpy(self, concentration, temperature):
        """Function computes the mixture reference enthalpy (cal/mol)"""
        species_enthalpy = np.zeros(self.num_species)
        for ii, ispecies in enumerate(concentration.keys()):
            species_enthalpy[ii] = self.compute_species_enthalpy(
                species_name=ispecies,
                temperature=temperature
            )

        mixture_enthalpy = self.weight_with_mole_fraction(
            concentration=concentration, quantity=species_enthalpy)

        return mixture_enthalpy

    def compute_mixture_internal_energy(self, concentration, temperature):
        """Function comptues the mixture internal energy (cal/mol)"""
        species_internal_energy = np.zeros(self.num_species)
        for ii, ispecies in enumerate(concentration.keys()):
            species_internal_energy[ii] = self.compute_species_internal_energy(
                species_name=ispecies,
                temperature=temperature
            )

        mixture_internal_energy = self.weight_with_mole_fraction(
            concentration=concentration, quantity=species_internal_energy)

        return mixture_internal_energy

    def compute_reaction_rates(self, concentration, activation_energy, a_1):
        """Function computes the reaction rates given the concentration, activation energy (E),
        and parameter a_1"""
        # Extractions
        n_dodecane_conc = concentration['C12H26']
        oxygen_conc = concentration['O2']
        carbon_monoxide_conc = concentration['CO']
        water_conc = concentration['H2O']
        carbon_dioxide_conc = concentration['CO2']

        A = self.compute_pre_exponential_factor(a_1=a_1)
        activation_energy_joules = activation_energy*4.2

    def compute_pre_exponential_factor(self, a_1):
        """Function computes the pre_exponential_factor (A)"""
        a_2 = self.pre_exponential_factor_parameters[0]
        a_3 = self.pre_exponential_factor_parameters[1]
        a_4 = self.pre_exponential_factor_parameters[2]
        a_5 = self.pre_exponential_factor_parameters[3]
        a_6 = self.pre_exponential_factor_parameters[4]
        a_7 = self.pre_exponential_factor_parameters[5]

        log_A = a_1 + a_2*np.exp(a_3*self.equivalence_ratio) + a_4 * \
            np.tanh((a_5+a_6*self.equivalence_ratio)*self.temp_init + a_7)

        return np.exp(log_A)


def main():

    # Units,
    # Pressure : Pa,
    # temperature : K,
    # Molar mass : kg/mol
    # Activation energy: cal/mol
    # Reference enthalpy (Enthalpy at 298.15K): cal/mol     (For C12H26 rmg mit database was used)

    with open("nasa_coeff.yml") as fh:
        nasa_polynomials = yaml.load(fh, Loader=yaml.FullLoader)

    # Pre-exponential factor (A) parameters
    pre_exponential_factor_parameters = [-2.13, -
                                         2.05, 1.89, -0.01, 2.87*10**-4, 8.43]

    model = ignition(
        pressure=101325,
        temp_init=300,
        num_species=5,
        species_mole_fraction_init={'C12H26': 1,
                                    'O2': 12.5,
                                    'CO': 0,
                                    'H2O': 0,
                                    'CO2': 0},
        species_molar_mass={'C12H26': 170.3398*10**-3,
                            'O2': 31.9989*10**-3,
                            'CO': 28.0101*10**-3,
                            'H2O': 18.01528*10**-3,
                            'CO2': 44.0095*10**-3},
        nasa_polynomials=nasa_polynomials,
        equivalence_ratio=0.5,
        pre_exponential_factor_parameters=pre_exponential_factor_parameters,
        species_reference_enthalpy={'C12H26': -69720,
                                    'O2': 0.0,
                                    'CO': -26317.8571,
                                    'H2O': -57577.6190,
                                    'CO2': -93692.8571},
    )


if __name__ == "__main__":
    main()
