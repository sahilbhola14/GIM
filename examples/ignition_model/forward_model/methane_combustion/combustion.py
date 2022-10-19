import cantera as ct
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import sys
import time


plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=25)
plt.rc("lines", linewidth=3)
plt.rc("axes", labelpad=18, titlepad=20)
plt.rc("legend", fontsize=18, framealpha=1.0)
plt.rc("xtick.major", size=6)
plt.rc("xtick.minor", size=4)
plt.rc("ytick.major", size=6)
plt.rc("ytick.minor", size=4)


class mech_1S_CH4_MP1():
    def __init__(self, initial_temperature, initial_pressure, equivalence_ratio, Arrhenius_A=1.1e+10, Arrhenius_Ea=20000, Arrhenius_a=1.0, Arrhenius_b=0.5):
        self.initial_temperature = initial_temperature
        self.initial_pressure = initial_pressure
        self.equivalence_ratio = equivalence_ratio

        # Arrhenius parameters (Reaction 1)
        self.Arrhenius_A = Arrhenius_A
        self.Arrhenius_Ea = Arrhenius_Ea
        self.Arrhenius_a = Arrhenius_a
        self.Arrhenius_b = Arrhenius_b

    def mech_1S_CH4_MP1_combustion(self):
        """Function computes the temperature profile for 1S_CH4_MP1 mechanism
        references: 
        https://www.cerfacs.fr/cantera/mechanisms/meth.php
        https://www.cerfacs.fr/cantera/docs/mechanisms/methane-air/GLOB/CANTERA/1S_CH4_MP1.cti
        """
        gas = ct.Solution(yaml = self.get_1S_CH4_MP1_gas_file())
        gas.TP = self.initial_temperature, self.initial_pressure
        gas.set_equivalence_ratio(self.equivalence_ratio, "CH4", "O2:1, N2:3.76")

        dt = 1e-6
        t_end = 3*1e-3

        reactor = ct.IdealGasConstPressureReactor(gas)
        sim = ct.ReactorNet([reactor])
        states = ct.SolutionArray(gas, extra=['t'])
        sim.atol = 1e-15
        sim.rtol = 1e-6

        for t in np.arange(0, t_end, dt):
            # print("Time : {0:.5f} [s]".format(t))
            sim.advance(t)
            states.append(reactor.thermo.state, t=t)

        return states

    def get_1S_CH4_MP1_gas_file(self):
        """Function returns the gas file for 1S_CH4_MP1 mechanism"""

        gas_file = '''

            description: |-
                Single step mechanism (1S CH4 MP1)

            units: {length: cm, time: s, quantity: mol, activation-energy: cal/mol}

            phases:
            - name: gas
              thermo: ideal-gas
              elements: [O, H, C, N, Ar]
              species: [O2, H2O, CH4, CO2, N2]
              kinetics: gas
              transport: mixture-averaged
              state: {T: 300.0, P: 1 atm}

            species:
            - name: O2
              composition: {O: 2}
              thermo:
                model: NASA7
                temperature-ranges: [200.0, 1000.0, 3500.0]
                data:
                - [3.78245636, -2.99673416e-03, 9.84730201e-06, -9.68129509e-09, 3.24372837e-12,
                  -1063.94356, 3.65767573]
                - [3.28253784, 1.48308754e-03, -7.57966669e-07, 2.09470555e-10, -2.16717794e-14,
                  -1088.45772, 5.45323129]
                note: TPIS89
              transport:
                model: gas
                geometry: linear
                well-depth: 107.4
                diameter: 3.458
                polarizability: 1.6
                rotational-relaxation: 3.8

            - name: H2O
              composition: {H: 2, O: 1}
              thermo:
                model: NASA7
                temperature-ranges: [200.0, 1000.0, 3500.0]
                data:
                - [4.19864056, -2.0364341e-03, 6.52040211e-06, -5.48797062e-09, 1.77197817e-12,
                  -3.02937267e+04, -0.849032208]
                - [3.03399249, 2.17691804e-03, -1.64072518e-07, -9.7041987e-11, 1.68200992e-14,
                  -3.00042971e+04, 4.9667701]
                note: L8/89
              transport:
                model: gas
                geometry: nonlinear
                well-depth: 572.4
                diameter: 2.605
                dipole: 1.844
                rotational-relaxation: 4.0

            - name: CH4
              composition: {C: 1, H: 4}
              thermo:
                model: NASA7
                temperature-ranges: [200.0, 1000.0, 3500.0]
                data:
                - [5.14987613, -0.0136709788, 4.91800599e-05, -4.84743026e-08, 1.66693956e-11,
                  -1.02466476e+04, -4.64130376]
                - [0.074851495, 0.0133909467, -5.73285809e-06, 1.22292535e-09, -1.0181523e-13,
                  -9468.34459, 18.437318]
                note: L8/88
              transport:
                model: gas
                geometry: nonlinear
                well-depth: 141.4
                diameter: 3.746
                polarizability: 2.6
                rotational-relaxation: 13.0

            - name: CO2
              composition: {C: 1, O: 2}
              thermo:
                model: NASA7
                temperature-ranges: [200.0, 1000.0, 3500.0]
                data:
                - [2.35677352, 8.98459677e-03, -7.12356269e-06, 2.45919022e-09, -1.43699548e-13,
                  -4.83719697e+04, 9.90105222]
                - [3.85746029, 4.41437026e-03, -2.21481404e-06, 5.23490188e-10, -4.72084164e-14,
                  -4.8759166e+04, 2.27163806]
                note: L7/88
              transport:
                model: gas
                geometry: linear
                well-depth: 244.0
                diameter: 3.763
                polarizability: 2.65
                rotational-relaxation: 2.1

            - name: N2
              composition: {N: 2}
              thermo:
                model: NASA7
                temperature-ranges: [300.0, 1000.0, 5000.0]
                data:
                - [3.298677, 1.4082404e-03, -3.963222e-06, 5.641515e-09, -2.444854e-12,
                  -1020.8999, 3.950372]
                - [2.92664, 1.4879768e-03, -5.68476e-07, 1.0097038e-10, -6.753351e-15,
                  -922.7977, 5.980528]
                note: '121286'
              transport:
                model: gas
                geometry: linear
                well-depth: 97.53
                diameter: 3.621
                polarizability: 1.76
                rotational-relaxation: 4.0

            reactions:
            - equation: CH4 + 2 O2 => CO2 + 2 H2O
              rate-constant: {A: %f, b: 0, Ea: %f}
              orders: {CH4: %f, O2: %f}
        '''%(self.Arrhenius_A, self.Arrhenius_Ea, self.Arrhenius_a, self.Arrhenius_b)

        return gas_file

    def compute_adiabatic_temperature(self):
        """Function computes the adiabatic temperature"""
        gas = ct.Solution(yaml = self.get_1S_CH4_MP1_gas_file())
        gas.TP = self.initial_temperature, self.initial_pressure
        gas.set_equivalence_ratio(self.equivalence_ratio, "CH4", "O2:1, N2:3.76")
        gas.equilibrate("HP")
        return gas.T

    def compute_equilibrate_species_concentration(self, species_names):
        """Function computest the equilibrate species concentration"""
        gas = ct.Solution(yaml = self.get_1S_CH4_MP1_gas_file())
        gas.TP = self.initial_temperature, self.initial_pressure
        gas.set_equivalence_ratio(self.equivalence_ratio, "CH4", "O2:1, N2:3.76")
        gas.equilibrate("HP")

        gas.selected_species = species_names
        species_final_mole_fraction = gas.X

        return species_final_mole_fraction


class mech_gri():
    def __init__(self, initial_temperature, initial_pressure, equivalence_ratio):
        self.initial_temperature = initial_temperature
        self.initial_pressure = initial_pressure
        self.equivalence_ratio = equivalence_ratio

    def gri_combustion(self):
        """function computes the temperature profile"""
        gas = ct.Solution("gri30.yaml")
        gas.TP = self.initial_temperature, self.initial_pressure
        gas.set_equivalence_ratio(self.equivalence_ratio, "CH4", "O2:1, N2:3.76")
        dt = 1e-6
        t_end = 3*1e-3
        reactor = ct.IdealGasConstPressureReactor(gas)
        sim = ct.ReactorNet([reactor])
        states = ct.SolutionArray(gas, extra=['t'])
        sim.atol = 1e-15
        sim.rtol = 1e-6

        for t in np.arange(0, t_end, dt):
            # print("Time : {0:.5f} [s]".format(t))
            sim.advance(t)
            states.append(reactor.thermo.state, t=t)

        return states

    def compute_adiabatic_temperature(self):
        """Function computes the adiabatic temperature"""
        gas = ct.Solution("gri30.yaml")
        gas.TP = self.initial_temperature, self.initial_pressure
        gas.set_equivalence_ratio(self.equivalence_ratio, "CH4", "O2:1, N2:3.76")
        gas.equilibrate("HP")
        return gas.T

    def compute_equilibrate_species_concentration(self, species_names):
        """Function computest the equilibrate species concentration"""
        gas = ct.Solution("gri30.yaml")
        gas.TP = self.initial_temperature, self.initial_pressure
        gas.set_equivalence_ratio(self.equivalence_ratio, "CH4", "O2:1, N2:3.76")
        gas.equilibrate("HP")

        gas.selected_species = species_names
        species_final_mole_fraction = gas.X

        return species_final_mole_fraction

    def compute_ignition_time(self):
        """Function computes the ignition temperature"""
        gas = ct.Solution("gri30.yaml")
        gas.TP = self.initial_temperature, self.initial_pressure
        gas.set_equivalence_ratio(self.equivalence_ratio, "CH4", "O2:1, N2:3.76")
        dt = 1e-6
        t_end = 1e+5
        reactor = ct.IdealGasConstPressureReactor(gas)
        sim = ct.ReactorNet([reactor])
        states = ct.SolutionArray(gas, extra=['t'])
        sim.atol = 1e-15
        sim.rtol = 1e-6

        while sim.time < t_end:
            try:
                sim.step()
                states.append(reactor.thermo.state, t=sim.time)
            except:
                sim.atol = sim.atol*10

        ignition_time, ignition_temperature = compute_ignition_stats(temperature=states.T, time=states.t)

        return ignition_time


class mech_1S_CH4_Westbrook():
    def __init__(self, initial_temperature, initial_pressure, equivalence_ratio, Arrhenius_A=8.3e+5, Arrhenius_Ea=30000, Arrhenius_a=-0.3, Arrhenius_b=1.3):
        self.initial_temperature = initial_temperature
        self.initial_pressure = initial_pressure
        self.equivalence_ratio = equivalence_ratio

        # Arrhenius parameters (Reaction 1)
        self.Arrhenius_A = Arrhenius_A
        self.Arrhenius_Ea = Arrhenius_Ea
        self.Arrhenius_a = Arrhenius_a
        self.Arrhenius_b = Arrhenius_b

    def mech_1S_CH4_Westbrook_combustion(self):
        """Function computes the temperature profile for 1S_CH4_MP1 mechanism
        references: 
        """
        gas = ct.Solution(yaml=self.get_1S_CH4_Westbrook_gas_file())
        gas.TP = self.initial_temperature, self.initial_pressure
        gas.set_equivalence_ratio(self.equivalence_ratio, "CH4", "O2:1, N2:3.76")

        dt = 1e-6
        t_end = 3*1e-3

        reactor = ct.IdealGasConstPressureReactor(gas)
        sim = ct.ReactorNet([reactor])
        states = ct.SolutionArray(gas, extra=['t'])
        sim.atol = 1e-15
        sim.rtol = 1e-6

        for t in np.arange(0, t_end, dt):
            # print("Time : {0:.5f} [s]".format(t))
            sim.advance(t)
            states.append(reactor.thermo.state, t=t)

        return states

    def get_1S_CH4_Westbrook_gas_file(self):
        """Function returns the gas file for 1S_CH4_MP1 mechanism"""
        negative_orders_flag = False
        if ((self.Arrhenius_a or self.Arrhenius_b) < 0):
            negative_orders_flag=True

        gas_file = '''

            description: |-
                Single step mechanism (1S CH4 Westbrook)

            units: {length: cm, time: s, quantity: mol, activation-energy: cal/mol}

            phases:
            - name: gas
              thermo: ideal-gas
              elements: [O, H, C, N, Ar]
              species: [O2, H2O, CH4, CO2, N2]
              kinetics: gas
              transport: mixture-averaged
              state: {T: 300.0, P: 1 atm}

            species:
            - name: O2
              composition: {O: 2}
              thermo:
                model: NASA7
                temperature-ranges: [200.0, 1000.0, 3500.0]
                data:
                - [3.78245636, -2.99673416e-03, 9.84730201e-06, -9.68129509e-09, 3.24372837e-12,
                  -1063.94356, 3.65767573]
                - [3.28253784, 1.48308754e-03, -7.57966669e-07, 2.09470555e-10, -2.16717794e-14,
                  -1088.45772, 5.45323129]
                note: TPIS89
              transport:
                model: gas
                geometry: linear
                well-depth: 107.4
                diameter: 3.458
                polarizability: 1.6
                rotational-relaxation: 3.8

            - name: H2O
              composition: {H: 2, O: 1}
              thermo:
                model: NASA7
                temperature-ranges: [200.0, 1000.0, 3500.0]
                data:
                - [4.19864056, -2.0364341e-03, 6.52040211e-06, -5.48797062e-09, 1.77197817e-12,
                  -3.02937267e+04, -0.849032208]
                - [3.03399249, 2.17691804e-03, -1.64072518e-07, -9.7041987e-11, 1.68200992e-14,
                  -3.00042971e+04, 4.9667701]
                note: L8/89
              transport:
                model: gas
                geometry: nonlinear
                well-depth: 572.4
                diameter: 2.605
                dipole: 1.844
                rotational-relaxation: 4.0

            - name: CH4
              composition: {C: 1, H: 4}
              thermo:
                model: NASA7
                temperature-ranges: [200.0, 1000.0, 3500.0]
                data:
                - [5.14987613, -0.0136709788, 4.91800599e-05, -4.84743026e-08, 1.66693956e-11,
                  -1.02466476e+04, -4.64130376]
                - [0.074851495, 0.0133909467, -5.73285809e-06, 1.22292535e-09, -1.0181523e-13,
                  -9468.34459, 18.437318]
                note: L8/88
              transport:
                model: gas
                geometry: nonlinear
                well-depth: 141.4
                diameter: 3.746
                polarizability: 2.6
                rotational-relaxation: 13.0

            - name: CO2
              composition: {C: 1, O: 2}
              thermo:
                model: NASA7
                temperature-ranges: [200.0, 1000.0, 3500.0]
                data:
                - [2.35677352, 8.98459677e-03, -7.12356269e-06, 2.45919022e-09, -1.43699548e-13,
                  -4.83719697e+04, 9.90105222]
                - [3.85746029, 4.41437026e-03, -2.21481404e-06, 5.23490188e-10, -4.72084164e-14,
                  -4.8759166e+04, 2.27163806]
                note: L7/88
              transport:
                model: gas
                geometry: linear
                well-depth: 244.0
                diameter: 3.763
                polarizability: 2.65
                rotational-relaxation: 2.1

            - name: N2
              composition: {N: 2}
              thermo:
                model: NASA7
                temperature-ranges: [300.0, 1000.0, 5000.0]
                data:
                - [3.298677, 1.4082404e-03, -3.963222e-06, 5.641515e-09, -2.444854e-12,
                  -1020.8999, 3.950372]
                - [2.92664, 1.4879768e-03, -5.68476e-07, 1.0097038e-10, -6.753351e-15,
                  -922.7977, 5.980528]
                note: '121286'
              transport:
                model: gas
                geometry: linear
                well-depth: 97.53
                diameter: 3.621
                polarizability: 1.76
                rotational-relaxation: 4.0

            reactions:
            - equation: CH4 + 2 O2 => CO2 + 2 H2O
              rate-constant: {A: %f, b: 0, Ea: %f}
              orders: {CH4: %f, O2: %f}
              negative-orders: %s 
        '''%(self.Arrhenius_A, self.Arrhenius_Ea, self.Arrhenius_a, self.Arrhenius_b, negative_orders_flag)
        return gas_file

    def compute_adiabatic_temperature(self):
        """Function computes the adiabatic temperature"""
        gas = ct.Solution(yaml = self.get_1S_CH4_Westbrook_gas_file())
        gas.TP = self.initial_temperature, self.initial_pressure
        gas.set_equivalence_ratio(self.equivalence_ratio, "CH4", "O2:1, N2:3.76")
        gas.equilibrate("HP")
        return gas.T

    def compute_equilibrate_species_concentration(self, species_names):
        """Function computest the equilibrate species concentration"""
        gas = ct.Solution(yaml = self.get_1S_CH4_Westbrook_gas_file())
        gas.TP = self.initial_temperature, self.initial_pressure
        gas.set_equivalence_ratio(self.equivalence_ratio, "CH4", "O2:1, N2:3.76")
        gas.equilibrate("HP")

        gas.selected_species = species_names
        species_final_mole_fraction = gas.X

        return species_final_mole_fraction

    def compute_ignition_time(self):
        """Function computes the ignition temperature"""
        gas = ct.Solution(yaml = self.get_1S_CH4_Westbrook_gas_file())
        gas.TP = self.initial_temperature, self.initial_pressure
        gas.set_equivalence_ratio(self.equivalence_ratio, "CH4", "O2:1, N2:3.76")
        dt = 1e-6
        t_end = 2000*1e-3
        reactor = ct.IdealGasConstPressureReactor(gas)
        sim = ct.ReactorNet([reactor])
        states = ct.SolutionArray(gas, extra=['t'])
        sim.atol = 1e-15
        sim.rtol = 1e-6

        while sim.time < t_end:
            sim.step()
            states.append(reactor.thermo.state, t=sim.time)

        ignition_time, ignition_temperature = compute_ignition_stats(temperature=states.T, time=states.t)

        return ignition_time


class mech_2S_CH4_Westbrook():
    def __init__(self, initial_temperature, initial_pressure, equivalence_ratio, Arrhenius_A=2.8e+9, Arrhenius_Ea=48400, Arrhenius_a=-0.3, Arrhenius_b=1.3):
        self.initial_temperature = initial_temperature
        self.initial_pressure = initial_pressure
        self.equivalence_ratio = equivalence_ratio

        # Arrhenius parameters (Reaction 1)
        self.Arrhenius_A = Arrhenius_A
        self.Arrhenius_Ea = Arrhenius_Ea
        self.Arrhenius_a = Arrhenius_a
        self.Arrhenius_b = Arrhenius_b

    def mech_2S_CH4_Westbrook_combustion(self):
        """Function computes the temperature profile for 1S_CH4_MP1 mechanism
        references: 
        """
        gas = ct.Solution(yaml=self.get_2S_CH4_Westbrook_gas_file())
        gas.TP = self.initial_temperature, self.initial_pressure
        gas.set_equivalence_ratio(self.equivalence_ratio, "CH4", "O2:1, N2:3.76")
        dt = 1e-6
        t_end = 3*1e-3
        reactor = ct.IdealGasConstPressureReactor(gas)
        sim = ct.ReactorNet([reactor])
        states = ct.SolutionArray(gas, extra=['t'])
        sim.atol = 1e-15
        sim.rtol = 1e-6
        for t in np.arange(0, t_end, dt):
            # print("Time : {0:.5f} [s]".format(t))
            sim.advance(t)
            states.append(reactor.thermo.state, t=t)
        return states

    def get_2S_CH4_Westbrook_gas_file(self):
        """Function returns the gas file for 1S_CH4_MP1 mechanism"""

        gas_file = '''

            description: |-
                Two-step mechanism (2S CH4 Westbrook)

            units: {length: cm, time: s, quantity: mol, activation-energy: cal/mol}

            phases:
            - name: gas
              thermo: ideal-gas
              elements: [O, H, C, N, Ar]
              species: [O2, H2O, CH4, CO2, N2, CO]
              kinetics: gas
              transport: mixture-averaged
              state: {T: 300.0, P: 1 atm}

            species:
            - name: O2
              composition: {O: 2}
              thermo:
                model: NASA7
                temperature-ranges: [200.0, 1000.0, 3500.0]
                data:
                - [3.78245636, -2.99673416e-03, 9.84730201e-06, -9.68129509e-09, 3.24372837e-12,
                  -1063.94356, 3.65767573]
                - [3.28253784, 1.48308754e-03, -7.57966669e-07, 2.09470555e-10, -2.16717794e-14,
                  -1088.45772, 5.45323129]
                note: TPIS89
              transport:
                model: gas
                geometry: linear
                well-depth: 107.4
                diameter: 3.458
                polarizability: 1.6
                rotational-relaxation: 3.8

            - name: H2O
              composition: {H: 2, O: 1}
              thermo:
                model: NASA7
                temperature-ranges: [200.0, 1000.0, 3500.0]
                data:
                - [4.19864056, -2.0364341e-03, 6.52040211e-06, -5.48797062e-09, 1.77197817e-12,
                  -3.02937267e+04, -0.849032208]
                - [3.03399249, 2.17691804e-03, -1.64072518e-07, -9.7041987e-11, 1.68200992e-14,
                  -3.00042971e+04, 4.9667701]
                note: L8/89
              transport:
                model: gas
                geometry: nonlinear
                well-depth: 572.4
                diameter: 2.605
                dipole: 1.844
                rotational-relaxation: 4.0

            - name: CH4
              composition: {C: 1, H: 4}
              thermo:
                model: NASA7
                temperature-ranges: [200.0, 1000.0, 3500.0]
                data:
                - [5.14987613, -0.0136709788, 4.91800599e-05, -4.84743026e-08, 1.66693956e-11,
                  -1.02466476e+04, -4.64130376]
                - [0.074851495, 0.0133909467, -5.73285809e-06, 1.22292535e-09, -1.0181523e-13,
                  -9468.34459, 18.437318]
                note: L8/88
              transport:
                model: gas
                geometry: nonlinear
                well-depth: 141.4
                diameter: 3.746
                polarizability: 2.6
                rotational-relaxation: 13.0

            - name: CO2
              composition: {C: 1, O: 2}
              thermo:
                model: NASA7
                temperature-ranges: [200.0, 1000.0, 3500.0]
                data:
                - [2.35677352, 8.98459677e-03, -7.12356269e-06, 2.45919022e-09, -1.43699548e-13,
                  -4.83719697e+04, 9.90105222]
                - [3.85746029, 4.41437026e-03, -2.21481404e-06, 5.23490188e-10, -4.72084164e-14,
                  -4.8759166e+04, 2.27163806]
                note: L7/88
              transport:
                model: gas
                geometry: linear
                well-depth: 244.0
                diameter: 3.763
                polarizability: 2.65
                rotational-relaxation: 2.1

            - name: N2
              composition: {N: 2}
              thermo:
                model: NASA7
                temperature-ranges: [300.0, 1000.0, 5000.0]
                data:
                - [3.298677, 1.4082404e-03, -3.963222e-06, 5.641515e-09, -2.444854e-12,
                  -1020.8999, 3.950372]
                - [2.92664, 1.4879768e-03, -5.68476e-07, 1.0097038e-10, -6.753351e-15,
                  -922.7977, 5.980528]
                note: '121286'
              transport:
                model: gas
                geometry: linear
                well-depth: 97.53
                diameter: 3.621
                polarizability: 1.76
                rotational-relaxation: 4.0

            - name: CO
              composition: {C: 1, O: 1}
              thermo:
                model: NASA7
                temperature-ranges: [200.0, 1000.0, 3500.0]
                data:
                - [3.57953347, -6.1035368e-04, 1.01681433e-06, 9.07005884e-10, -9.04424499e-13,
                  -1.4344086e+04, 3.50840928]
                - [2.71518561, 2.06252743e-03, -9.98825771e-07, 2.30053008e-10, -2.03647716e-14,
                  -1.41518724e+04, 7.81868772]
                note: TPIS79
              transport:
                model: gas
                geometry: linear
                well-depth: 98.1
                diameter: 3.65
                polarizability: 1.95
                rotational-relaxation: 1.8

            reactions:
            - equation: CH4 + 1.5 O2 => CO + 2 H2O  # Reaction 1
              rate-constant: {A: %f, b: 0 , Ea: %f}
              orders: {CH4: %f, O2: %f}
              negative-orders: true
            - equation: CO + 0.5 O2 => CO2 # Reaction 2
              rate-constant: {A: 3.98e+14, b: 0 , Ea: 40000.0}
              orders: {CO: 1, H2O: 0.5, O2: 0.25}
              nonreactant-orders: true
            - equation: CO2 => CO + 0.5 O2 # Reaction 3
              rate-constant: {A: 5.0e+8, b: 0, Ea: 40000.0}
              orders: {CO2: 1}
        '''%(self.Arrhenius_A, self.Arrhenius_Ea, self.Arrhenius_a, self.Arrhenius_b)

        return gas_file

    def compute_adiabatic_temperature(self):
        """Function computes the adiabatic temperature"""
        gas = ct.Solution(yaml = self.get_2S_CH4_Westbrook_gas_file())
        gas.TP = self.initial_temperature, self.initial_pressure
        gas.set_equivalence_ratio(self.equivalence_ratio, "CH4", "O2:1, N2:3.76")
        gas.equilibrate("HP")
        return gas.T

    def compute_equilibrate_species_concentration(self, species_names):
        """Function computest the equilibrate species concentration"""
        gas = ct.Solution(yaml = self.get_2S_CH4_Westbrook_gas_file())
        gas.TP = self.initial_temperature, self.initial_pressure
        gas.set_equivalence_ratio(self.equivalence_ratio, "CH4", "O2:1, N2:3.76")
        gas.equilibrate("HP")

        gas.selected_species = species_names
        species_final_mole_fraction = gas.X

        return species_final_mole_fraction

    def compute_ignition_time(self):
        """Function computes the ignition temperature"""
        gas = ct.Solution(yaml = self.get_2S_CH4_Westbrook_gas_file())
        gas.TP = self.initial_temperature, self.initial_pressure
        gas.set_equivalence_ratio(self.equivalence_ratio, "CH4", "O2:1, N2:3.76")
        dt = 1e-6
        t_end = 1e+5
        reactor = ct.IdealGasConstPressureReactor(gas)
        sim = ct.ReactorNet([reactor])
        states = ct.SolutionArray(gas, extra=['t'])
        sim.rtol = 1e-8
        simulation_success = False

        while sim.time < t_end:
            try:
                sim.step()
                states.append(reactor.thermo.state, t=sim.time)
                simulation_success = True
            except:
                sim.atol = sim.atol*10
                simulation_success = False
                if sim.atol > 1e-1:
                    break
        if simulation_success:
            ignition_time, ignition_temperature = compute_ignition_stats(temperature=states.T, time=states.t)
        else:
            ignition_time = t_end

        return ignition_time


def compute_ignition_stats(temperature, time):
    dt_array = time[1:] - time[0:-1]
    temperature_derivative = np.zeros_like(temperature)
    temperature_derivative[0] = (temperature[1] - temperature[0]) / dt_array[0]
    temperature_derivative[-1] = (temperature[-1] - temperature[-2]) / dt_array[-2]
    temperature_derivative[1:-1] = (temperature[2:] - temperature[0:-2]) / (dt_array[1:] + dt_array[:-1])
    ignition_temperature = temperature[ np.argmax(temperature_derivative) ]
    ignition_time = time[ np.argmax(temperature_derivative) ]

    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.xscale("log")
    # plt.plot(time, temperature)
    # plt.subplot(1, 2, 2)
    # plt.plot(time, temperature_derivative)
    # plt.xscale("log")
    # plt.show()
    # breakpoint()

    return ignition_time, ignition_temperature

def main():
    # Begin user input
    initial_temperature = 300 # K
    initial_pressure = 100000 # Pa
    equivalence_ratio = 1.0 # For transient cases
    eval_equivalence_ratio_adiabatic = np.linspace(0.5, 1.5, 100) # For equilibrium studies
    eval_auto_ignition_temp = np.linspace(800, 2000, 5)
    # eval_auto_ignition_temp = np.array([2500])
    # End user input

    ## GRIMech3.0 combustion model
    # gri_eval_species_name = ['CO2', 'CO', 'H2O', 'H2', 'CH4', 'N2', 'O2']
    # tic = time.time()
    # gri_model = mech_gri(initial_temperature, initial_pressure, equivalence_ratio)
    # gri_states = gri_model.gri_combustion()
    # toc = time.time()
    # print("Compute time GRIMech3.0 : {0:.5f} [s]".format(toc-tic))

    # # Compute the adiabatic flame temperature
    # gri_species_concentration = np.zeros((len(gri_eval_species_name), eval_equivalence_ratio_adiabatic.shape[0]))
    # gri_adiabatic_temp = np.zeros_like(eval_equivalence_ratio_adiabatic)
    # for ii, iratio in enumerate(eval_equivalence_ratio_adiabatic):
    #     gri_model = mech_gri(initial_temperature=initial_temperature, initial_pressure=initial_pressure, equivalence_ratio=iratio)
    #     gri_adiabatic_temp[ii] = gri_model.compute_adiabatic_temperature()
    #     gri_species_concentration[:, ii] = gri_model.compute_equilibrate_species_concentration(species_names=gri_eval_species_name)
    # gri_species_dict = {}
    # for ispecies in range(len(gri_eval_species_name)):
    #     gri_species_dict[gri_eval_species_name[ispecies]] = gri_species_concentration[ispecies, :]

    # # Computing the ignition temerature 
    gri_ignition_time = np.zeros((4, eval_auto_ignition_temp.shape[0]))
    gri_ignition_time[0, :] = eval_auto_ignition_temp
    gri_ignition_time[1, :] = initial_pressure
    gri_ignition_time[2, :] = np.ones(eval_auto_ignition_temp.shape[0])
    tic = time.time()
    for ii, itemp in enumerate(eval_auto_ignition_temp):
        gri_model = mech_gri(initial_temperature=itemp, initial_pressure=initial_pressure, equivalence_ratio=1.0)
        gri_ignition_time[3, ii] = gri_model.compute_ignition_time()
    np.savetxt("./data/ignition_time/ignition_time_gri_mech.dat", gri_ignition_time.T, delimiter=' ', header = "Vartiables: Inital_temperature, Initial_pressure, Equivalence_ratio, Ignition_temperature")
    print("Ignition time (GRI) : {}".format(time.time() - tic))

    # # 1S_CH4_MP1
    # mech_1S_CH4_MP1_eval_species_name = ['CO2', 'H2O','CH4', 'N2', 'O2']
    # tic = time.time()
    # model_1S_CH4_MP1 = mech_1S_CH4_MP1(initial_temperature, initial_pressure, equivalence_ratio)
    # mech_1S_CH4_MP1_states = model_1S_CH4_MP1.mech_1S_CH4_MP1_combustion()
    # toc = time.time()
    # print("Compute time 1S_CH4_MP1 : {0:.5f} [s]".format(toc-tic))
    # mech_1S_CH4_MP1_species_concentration = np.zeros((len(mech_1S_CH4_MP1_eval_species_name), eval_equivalence_ratio_adiabatic.shape[0]))
    # mech_1S_CH4_MP1_adiabatic_temp= np.zeros_like(eval_equivalence_ratio_adiabatic)
    # for ii, iratio in enumerate(eval_equivalence_ratio_adiabatic):
    #     model_1S_CH4_MP1 = mech_1S_CH4_MP1(initial_temperature=300, initial_pressure=100000, equivalence_ratio=iratio)
    #     mech_1S_CH4_MP1_adiabatic_temp[ii] = model_1S_CH4_MP1.compute_adiabatic_temperature()
    #     mech_1S_CH4_MP1_species_concentration[:, ii] = model_1S_CH4_MP1.compute_equilibrate_species_concentration(species_names=mech_1S_CH4_MP1_eval_species_name)
    # mech_1S_CH4_MP1_species_dict = {}
    # for ispecies in range(len(mech_1S_CH4_MP1_eval_species_name)):
    #     mech_1S_CH4_MP1_species_dict[mech_1S_CH4_MP1_eval_species_name[ispecies]] = mech_1S_CH4_MP1_species_concentration[ispecies, :]


    # # 1S_CH4_Westbrook
    # mech_1S_CH4_Westbrook_eval_species_name = ['CO2', 'H2O','CH4', 'N2', 'O2']
    # tic = time.time()
    # model_1S_CH4_Westbrook = mech_1S_CH4_Westbrook(initial_temperature, initial_pressure, equivalence_ratio)
    # mech_1S_CH4_Westbrook_states = model_1S_CH4_Westbrook.mech_1S_CH4_Westbrook_combustion()
    # toc = time.time()
    # print("Compute time 1S_CH4_Westbrook : {0:.5f} [s]".format(toc-tic))
    # mech_1S_CH4_Westbrook_species_concentration = np.zeros((len(mech_1S_CH4_Westbrook_eval_species_name), eval_equivalence_ratio_adiabatic.shape[0]))
    # mech_1S_CH4_Westbrook_adiabatic_temp= np.zeros_like(eval_equivalence_ratio_adiabatic)
    # for ii, iratio in enumerate(eval_equivalence_ratio_adiabatic):
    #     model_1S_CH4_Westbrook = mech_1S_CH4_Westbrook(initial_temperature=initial_temperature, initial_pressure=initial_pressure, equivalence_ratio=iratio)
    #     mech_1S_CH4_Westbrook_adiabatic_temp[ii] = model_1S_CH4_Westbrook.compute_adiabatic_temperature()
    #     mech_1S_CH4_Westbrook_species_concentration[:, ii] = model_1S_CH4_Westbrook.compute_equilibrate_species_concentration(species_names=mech_1S_CH4_Westbrook_eval_species_name)
    # mech_1S_CH4_Westbrook_species_dict = {}
    # for ispecies in range(len(mech_1S_CH4_Westbrook_eval_species_name)):
    #     mech_1S_CH4_Westbrook_species_dict[mech_1S_CH4_Westbrook_eval_species_name[ispecies]] = mech_1S_CH4_Westbrook_species_concentration[ispecies, :]

    # # Computing the ignition temerature 
    # mech_1S_CH4_Westbrook_ignition_time = np.zeros((4, eval_auto_ignition_temp.shape[0]))
    # mech_1S_CH4_Westbrook_ignition_time[0, :] = eval_auto_ignition_temp
    # mech_1S_CH4_Westbrook_ignition_time[1, :] = initial_pressure
    # mech_1S_CH4_Westbrook_ignition_time[2, :] = np.ones(eval_auto_ignition_temp.shape[0])
    # tic = time.time()
    # for ii, itemp in enumerate(eval_auto_ignition_temp):
    #     model_1S_CH4_Westbrook = mech_1S_CH4_Westbrook(initial_temperature=itemp, initial_pressure=initial_pressure, equivalence_ratio=1.0)
    #     mech_1S_CH4_Westbrook_ignition_time[3, ii] = model_1S_CH4_Westbrook.compute_ignition_time()
    # np.savetxt("./data/ignition_time/ignition_time_1S_CH4_Westbrook_mech.dat", mech_1S_CH4_Westbrook_ignition_time.T, delimiter=' ', header = "Vartiables: Inital_temperature, Initial_pressure, Equivalence_ratio, Ignition_temperature")
    # print("Ignition time computation (1S Westbrook) : {}".format(time.time() - tic))

    # # 2S_CH4_Westbrook
    # mech_2S_CH4_Westbrook_eval_species_name = ['CO2', 'H2O','CH4', 'N2', 'O2', 'CO']
    # tic = time.time()
    # model_2S_CH4_Westbrook = mech_2S_CH4_Westbrook(initial_temperature, initial_pressure, equivalence_ratio)
    # mech_2S_CH4_Westbrook_states = model_2S_CH4_Westbrook.mech_2S_CH4_Westbrook_combustion()
    # toc = time.time()
    # print("Compute time 2S_CH4_Westbrook : {0:.5f} [s]".format(toc-tic))
    # mech_2S_CH4_Westbrook_species_concentration = np.zeros((len(mech_2S_CH4_Westbrook_eval_species_name), eval_equivalence_ratio_adiabatic.shape[0]))
    # mech_2S_CH4_Westbrook_adiabatic_temp= np.zeros_like(eval_equivalence_ratio_adiabatic)
    # for ii, iratio in enumerate(eval_equivalence_ratio_adiabatic):
    #     model_2S_CH4_Westbrook = mech_2S_CH4_Westbrook(initial_temperature=initial_temperature, initial_pressure=initial_pressure, equivalence_ratio=iratio)
    #     mech_2S_CH4_Westbrook_adiabatic_temp[ii] = model_2S_CH4_Westbrook.compute_adiabatic_temperature()
    #     mech_2S_CH4_Westbrook_species_concentration[:, ii] = model_2S_CH4_Westbrook.compute_equilibrate_species_concentration(species_names=mech_2S_CH4_Westbrook_eval_species_name)

    # mech_2S_CH4_Westbrook_species_dict = {}
    # for ispecies in range(len(mech_2S_CH4_Westbrook_eval_species_name)):
    #     mech_2S_CH4_Westbrook_species_dict[mech_2S_CH4_Westbrook_eval_species_name[ispecies]] = mech_2S_CH4_Westbrook_species_concentration[ispecies, :]

    mech_2S_CH4_Westbrook_ignition_time = np.zeros((4, eval_auto_ignition_temp.shape[0]))
    mech_2S_CH4_Westbrook_ignition_time[0, :] = eval_auto_ignition_temp
    mech_2S_CH4_Westbrook_ignition_time[1, :] = initial_pressure
    mech_2S_CH4_Westbrook_ignition_time[2, :] = np.ones(eval_auto_ignition_temp.shape[0])

    tic = time.time()
    for ii, itemp in enumerate(eval_auto_ignition_temp):
        model_2S_CH4_Westbrook = mech_2S_CH4_Westbrook(initial_temperature=itemp, initial_pressure=initial_pressure, equivalence_ratio=1.0)
        mech_2S_CH4_Westbrook_ignition_time[3, ii] = model_2S_CH4_Westbrook.compute_ignition_time()
    print(mech_2S_CH4_Westbrook_ignition_time)
    np.savetxt("./data/ignition_time/ignition_time_2S_CH4_Westbrook_mech.dat", mech_2S_CH4_Westbrook_ignition_time.T, delimiter=' ', header = "Vartiables: Inital_temperature, Initial_pressure, Equivalence_ratio, Ignition_temperature")
    print("Ignition time computation (2S Westbrook) : {}".format(time.time() - tic))

    if '--ignitionplot' in sys.argv:
        fig, axs = plt.subplots(figsize=(10, 6))
        axs.scatter(1000/eval_auto_ignition_temp, mech_1S_CH4_Westbrook_ignition_time[1, :], label=r"1-step mechanism [Westbrook \textit{et al.}]", color="r", marker="^", s=60)
        axs.scatter(1000/eval_auto_ignition_temp, mech_2S_CH4_Westbrook_ignition_time[1, :], label=r"2-step mechanism [Westbrook \textit{el al.}]", color="b", marker="D", s=60)
        axs.scatter(1000/eval_auto_ignition_temp, gri_ignition_time[1, :], label="Gri-Mech 3.0", color="k", marker="s", s=60)
        axs.set_xlabel(r"1000/T [1/K]")
        axs.set_ylabel(r"$t_{ign}$ [s]")
        axs.legend(loc="upper left")
        axs.set_yscale("log")
        axs.grid(color="k", alpha=0.5)
        plt.tight_layout()
        plt.savefig("ignition_time_phi_{}_pressure_{}.png".format(1.0, initial_pressure))
        plt.close()

    if '--equilibrateplot' in sys.argv:
        fig, axs = plt.subplots(2, 2, figsize=(18, 10))
        fig.tight_layout(h_pad=5, w_pad=5)
        axs[0, 0].plot(eval_equivalence_ratio_adiabatic, mech_1S_CH4_Westbrook_adiabatic_temp, color="r",  label=r"1-step mechanism [Westbrook $\textit{et al.}$]")
        axs[0, 0].plot(eval_equivalence_ratio_adiabatic, mech_2S_CH4_Westbrook_adiabatic_temp, color="b",  label=r"2-step mechanism [Westbrook $\textit{et al.}$]")
        axs[0, 0].plot(eval_equivalence_ratio_adiabatic, gri_adiabatic_temp, color="k", label="Gri-Mech 3.0")
        axs[0, 0].set_xlabel(r"Equivalence ratio, $\phi$")
        axs[0, 0].set_ylabel(r"Adiabatic temperature [K]")
        axs[0, 0].set_ylim([1400, 2400])
        axs[0, 0].set_xlim([0.5, 1.5])
        axs[0, 0].tick_params(pad=10)
        # axs[0, 0].legend(loc="lower right")
        axs[0, 0].xaxis.set_major_locator(MultipleLocator(0.5))
        axs[0, 0].xaxis.set_minor_locator(MultipleLocator(0.25))
        axs[0, 0].yaxis.set_major_locator(MultipleLocator(200))
        axs[0, 0].grid(which="major", color="k", alpha=0.5)
        axs[0, 0].grid(which="minor", color="grey", alpha=0.3)

        axs[0, 1].plot(eval_equivalence_ratio_adiabatic, mech_1S_CH4_Westbrook_species_dict['CH4'], color="r", label=r"1-step mechanism [Westbrook $\textit{et al.}$]")
        axs[0, 1].plot(eval_equivalence_ratio_adiabatic, mech_2S_CH4_Westbrook_species_dict['CH4'], color="b", label=r"2-step mechanism [Westbrook $\textit{et al.}$]")
        axs[0, 1].plot(eval_equivalence_ratio_adiabatic, gri_species_dict['CH4'], color="k", label="Gri-Mech 3.0")
        axs[0, 1].set_xlim([0.5, 1.5])
        axs[0, 1].set_yscale("log")
        axs[0, 1].xaxis.set_major_locator(MultipleLocator(0.5))
        axs[0, 1].xaxis.set_minor_locator(MultipleLocator(0.25))
        axs[0, 1].grid(which="major", color="k", alpha=0.5)
        axs[0, 1].grid(which="minor", color="grey", alpha=0.3)
        axs[0, 1].set_xlabel(r"Equivalence ratio, $\phi$")
        axs[0, 1].set_ylabel(r"$[\textrm{CH}_{4}]$")
        # axs[0, 1].legend(loc="lower right")
        
        axs[1, 0].plot(eval_equivalence_ratio_adiabatic, mech_1S_CH4_Westbrook_species_dict['CO2'], color="r", label=r"1-step mechanism [Westbrook $\textit{et al.}$]")
        axs[1, 0].plot(eval_equivalence_ratio_adiabatic, mech_2S_CH4_Westbrook_species_dict['CO2'], color="b", label=r"2-step mechanism [Westbrook $\textit{et al.}$]")
        axs[1, 0].plot(eval_equivalence_ratio_adiabatic, gri_species_dict['CO2'], color="k", label="Gri-Mech 3.0")
        axs[1, 0].set_xlim([0.5, 1.5])
        axs[1, 0].set_ylim([0, 0.1])
        axs[1, 0].xaxis.set_major_locator(MultipleLocator(0.5))
        axs[1, 0].xaxis.set_minor_locator(MultipleLocator(0.25))
        axs[1, 0].yaxis.set_minor_locator(MultipleLocator(0.025))
        axs[1, 0].grid(which="major", color="k", alpha=0.5)
        axs[1, 0].grid(which="minor", color="grey", alpha=0.3)
        axs[1, 0].set_xlabel(r"Equivalence ratio, $\phi$")
        axs[1, 0].set_ylabel(r"$[\textrm{CO}_{2}]$")
        # axs[1, 0].legend(loc="lower right")
        
        # axs[1, 1].plot(eval_equivalence_ratio_adiabatic, mech_1S_CH4_Westbrook_species_dict['CO'], color="r", label=r"1-step mechanism [Westbrook $\textit{et al.}$]")
        axs[1, 1].plot(eval_equivalence_ratio_adiabatic, mech_2S_CH4_Westbrook_species_dict['CO'], color="b", label=r"2-step mechanism [Westbrook $\textit{et al.}$]")
        axs[1, 1].plot(eval_equivalence_ratio_adiabatic, gri_species_dict['CO'], color="k", label="Gri-Mech 3.0")
        axs[1, 1].set_xlim([0.5, 1.5])
        axs[1, 1].set_ylim([0, 0.15])
        axs[1, 1].xaxis.set_major_locator(MultipleLocator(0.5))
        axs[1, 1].xaxis.set_minor_locator(MultipleLocator(0.25))
        axs[1, 1].yaxis.set_minor_locator(MultipleLocator(0.025))
        axs[1, 1].grid(which="major", color="k", alpha=0.5)
        axs[1, 1].grid(which="minor", color="grey", alpha=0.3)
        axs[1, 1].set_xlabel(r"Equivalence ratio, $\phi$")
        axs[1, 1].set_ylabel(r"$[\textrm{CO}]$")
        # axs[1, 1].legend(loc="lower right")


        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.34, 0.5, 0.5, 0.5), ncol=3)

        plt.subplots_adjust(top=0.9, bottom=0.15, left=0.1)
        plt.savefig("equilibrate_initial_temp_{}_pressure_{}.png".format(initial_temperature, initial_pressure))
        plt.close()

    if '--transientplot' in sys.argv:
        fig, axs = plt.subplots(2, 2, figsize=(18, 10))
        fig.tight_layout(h_pad=5, w_pad=5)
        axs[0, 0].plot(gri_states.t*1000, gri_states.T, color="k", label="Gri-Mech3.0")
        axs[0, 0].plot(mech_1S_CH4_MP1_states.t*1000, mech_1S_CH4_MP1_states.T, color="r", label="1S_CH4_MP1")
        axs[0, 0].plot(mech_1S_CH4_Westbrook_states.t*1000, mech_1S_CH4_Westbrook_states.T, color="g", label="1S_CH4_Westbrook")
        axs[0, 0].plot(mech_2S_CH4_Westbrook_states.t*1000, mech_2S_CH4_Westbrook_states.T, color="b", label="2S_CH4_Westbrook")
        axs[0, 0].set_ylim([1400, 3500])
        # axs[0, 0].set_xlim([0.0000007, 0.003])
        axs[0, 0].set_xlabel("time [ms]")
        axs[0, 0].set_ylabel("Temperature [K]")
        # axs[0, 0].set_xscale("log")
        axs[0, 0].xaxis.set_major_locator(MultipleLocator(1))
        axs[0, 0].xaxis.set_minor_locator(MultipleLocator(0.5))
        axs[0, 0].grid()

        axs[0, 1].plot(gri_states.t*1000, gri_states('CH4').X, color="k", label="Gri-Mech3.0")
        axs[0, 1].plot(mech_1S_CH4_MP1_states.t*1000, mech_1S_CH4_MP1_states('CH4').X, color="r", label="1S_CH4_MP1")
        axs[0, 1].plot(mech_1S_CH4_Westbrook_states.t*1000, mech_1S_CH4_Westbrook_states('CH4').X, color="g", label="1S_CH4_Westbrook")
        axs[0, 1].plot(mech_2S_CH4_Westbrook_states.t*1000, mech_2S_CH4_Westbrook_states('CH4').X, color="b", label="2S_CH4_Westbrook")
        axs[0, 1].set_ylim([0, 0.1])
        # axs[0, 1].set_xlim([0.0000007, 0.003])
        axs[0, 1].set_xlabel("time [ms]")
        axs[0, 1].set_ylabel("[$CH_{4}$]")
        axs[0, 1].xaxis.set_major_locator(MultipleLocator(1))
        axs[0, 1].xaxis.set_minor_locator(MultipleLocator(0.5))
        # axs[0, 1].set_xscale("log")
        axs[0, 1].grid()

        axs[1, 0].plot(gri_states.t*1000, gri_states('O2').X, color="k", label="Gri-Mech3.0")
        axs[1, 0].plot(mech_1S_CH4_MP1_states.t*1000, mech_1S_CH4_MP1_states('O2').X, color="r", label="1S_CH4_MP1")
        axs[1, 0].plot(mech_1S_CH4_Westbrook_states.t*1000, mech_1S_CH4_Westbrook_states('O2').X, color="g", label="1S_CH4_Westbrook")
        axs[1, 0].plot(mech_2S_CH4_Westbrook_states.t*1000, mech_2S_CH4_Westbrook_states('O2').X, color="b", label="2S_CH4_Westbrook")
        axs[1, 0].set_ylim([0, 0.2])
        # axs[1, 0].set_xlim([0.0000007, 0.003])
        axs[1, 0].set_xlabel("time [ms]")
        axs[1, 0].set_ylabel("[$O_{2}$]")
        # axs[1, 0].set_xscale("log")
        axs[1, 0].xaxis.set_major_locator(MultipleLocator(1))
        axs[1, 0].xaxis.set_minor_locator(MultipleLocator(0.5))
        axs[1, 0].grid()

        axs[1, 1].plot(gri_states.t*1000, gri_states('CO2').X, color="k", label="Gri-Mech3.0")
        axs[1, 1].plot(mech_1S_CH4_MP1_states.t*1000, mech_1S_CH4_MP1_states('CO2').X, color="r", label="1S_CH4_MP1")
        axs[1, 1].plot(mech_1S_CH4_Westbrook_states.t*1000, mech_1S_CH4_Westbrook_states('CO2').X, color="g", label="1S_CH4_Westbrook")
        axs[1, 1].plot(mech_2S_CH4_Westbrook_states.t*1000, mech_2S_CH4_Westbrook_states('CO2').X, color="b", label="2S_CH4_Westbrook")
        axs[1, 1].set_ylim([0, 0.1])
        # axs[1, 1].set_xlim([0.0000007, 0.003])
        axs[1, 1].set_xlabel("time [ms]")
        axs[1, 1].set_ylabel("[$CO_{2}$]")
        # axs[1, 1].set_xscale("log")
        axs[1, 1].xaxis.set_major_locator(MultipleLocator(1))
        axs[1, 1].xaxis.set_minor_locator(MultipleLocator(0.5))
        axs[1, 1].grid()

        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.49, 0.5, 0.5, 0.5), ncol=4)
        plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15)
        plt.savefig("methane_combustion_t_{}_p_{}.png".format(initial_temperature, initial_pressure))
        plt.close()

if __name__ == ("__main__"):
    main()
