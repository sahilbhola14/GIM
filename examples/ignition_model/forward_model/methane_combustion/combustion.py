import cantera as ct
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import sys
import time

# Begin user input
initial_temp = 300
initial_pressure = 101325
# End user input


plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=25)
plt.rc("lines", linewidth=3)
plt.rc("axes", labelpad=18, titlepad=20)

def gri_combustion(initial_temperature, initial_pressure, equivalence_ratio):
    """function computes the temperature profile"""
    gas = ct.Solution("gri30.yaml")
    gas.TP = initial_temperature, initial_pressure
    gas.set_equivalence_ratio(equivalence_ratio, "CH4", "O2:1, N2:3.76")

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

def mech_1S_CH4_MP1(initial_temperature, initial_pressure, equivalence_ratio):
    """Function computes the temperature profile for 1S_CH4_MP1 mechanism
    references: 
    https://www.cerfacs.fr/cantera/mechanisms/meth.php
    https://www.cerfacs.fr/cantera/docs/mechanisms/methane-air/GLOB/CANTERA/1S_CH4_MP1.cti
    """
    gas = ct.Solution(yaml = get_1S_CH4_MP1_gas_file())
    gas.TP = initial_temperature, initial_pressure
    gas.set_equivalence_ratio(equivalence_ratio, "CH4", "O2:1, N2:3.76")

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

def mech_1S_CH4_Westbrook(initial_temperature, initial_pressure, equivalence_ratio):
    """Function computes the temperature profile for 1S_CH4_MP1 mechanism
    references: 
    """
    gas = ct.Solution(yaml=get_1S_CH4_Westbrook_gas_file())
    gas.TP = initial_temperature, initial_pressure
    gas.set_equivalence_ratio(equivalence_ratio, "CH4", "O2:1, N2:3.76")

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

def mech_2S_CH4_Westbrook(initial_temperature, initial_pressure, equivalence_ratio):
    """Function computes the temperature profile for 1S_CH4_MP1 mechanism
    references: 
    """
    gas = ct.Solution(yaml=get_2S_CH4_Westbrook_gas_file())
    gas.TP = initial_temperature, initial_pressure
    gas.set_equivalence_ratio(equivalence_ratio, "CH4", "O2:1, N2:3.76")

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

def get_1S_CH4_MP1_gas_file():
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
          rate-constant: {A: 1.1e+10, b: 0, Ea: 20000}
          orders: {CH4: 1.0, O2: 0.5}
    '''

    return gas_file

def get_1S_CH4_Westbrook_gas_file():
    """Function returns the gas file for 1S_CH4_MP1 mechanism"""

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
          rate-constant: {A: 8.3e+5, b: 0, Ea: 30000}
          orders: {CH4: -0.3, O2: 1.3}
          negative-orders: True
    '''

    return gas_file

def get_2S_CH4_Westbrook_gas_file():
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
          rate-constant: {A: 2.8e+9, b: 0 , Ea: 48400}
          orders: {CH4: -0.3, O2: 1.3}
          negative-orders: true
        - equation: CO + 0.5 O2 => CO2 # Reaction 2
          rate-constant: {A: 3.98e+14, b: 0 , Ea: 40000.0}
          orders: {CO: 1, H2O: 0.5, O2: 0.25}
          nonreactant-orders: true
        - equation: CO2 => CO + 0.5 O2 # Reaction 3
          rate-constant: {A: 5.0e+8, b: 0, Ea: 40000.0}
          orders: {CO2: 1}
    '''

    return gas_file

def main():
    # Begin user input
    initial_temperature = 1500 # K
    initial_pressure = 101325 # Pa
    equivalence_ratio = 1.0
    # End user input

    # GRIMech3.0 combustion model
    tic = time.time()
    gri_states = gri_combustion(initial_temperature, initial_pressure, equivalence_ratio)
    toc = time.time()
    print("Compute time GRIMech3.0 : {0:.5f} [s]".format(toc-tic))
    # 1S_CH4_MP1
    tic = time.time()
    mech_1S_CH4_MP1_states = mech_1S_CH4_MP1(initial_temperature, initial_pressure, equivalence_ratio)
    toc = time.time()
    print("Compute time 1S_CH4_MP1 : {0:.5f} [s]".format(toc-tic))
    # 1S_CH4_Westbrook
    tic = time.time()
    mech_1S_CH4_Westbrook_states = mech_1S_CH4_Westbrook(initial_temperature, initial_pressure, equivalence_ratio)
    toc = time.time()
    print("Compute time 1S_CH4_Westbrook : {0:.5f} [s]".format(toc-tic))
    # 2S_CH4_Westbrook
    tic = time.time()
    mech_2S_CH4_Westbrook_states = mech_2S_CH4_Westbrook(initial_temperature, initial_pressure, equivalence_ratio)
    toc = time.time()
    print("Compute time 2S_CH4_Westbrook : {0:.5f} [s]".format(toc-tic))
   
    if '--plot' in sys.argv:
        fig, axs = plt.subplots(2, 2, figsize=(18, 10))
        fig.tight_layout(h_pad=5, w_pad=5)
        axs[0, 0].plot(gri_states.t, gri_states.T, color="k", label="Gri-Mech3.0")
        axs[0, 0].plot(mech_1S_CH4_MP1_states.t, mech_1S_CH4_MP1_states.T, color="r", label="1S_CH4_MP1")
        axs[0, 0].plot(mech_1S_CH4_Westbrook_states.t, mech_1S_CH4_Westbrook_states.T, color="g", label="1S_CH4_Westbrook")
        axs[0, 0].plot(mech_2S_CH4_Westbrook_states.t, mech_2S_CH4_Westbrook_states.T, color="b", label="2S_CH4_Westbrook")
        axs[0, 0].set_ylim([1400, 3500])
        axs[0, 0].set_xlim([0.0000007, 0.003])
        axs[0, 0].set_xlabel("time [s]")
        axs[0, 0].set_ylabel("Temperature [K]")
        axs[0, 0].set_xscale("log")
        axs[0, 0].grid()
        # axs[0, 0].legend(loc="lower left")

        axs[0, 1].plot(gri_states.t, gri_states('CH4').X, color="k", label="Gri-Mech3.0")
        axs[0, 1].plot(mech_1S_CH4_MP1_states.t, mech_1S_CH4_MP1_states('CH4').X, color="r", label="1S_CH4_MP1")
        axs[0, 1].plot(mech_1S_CH4_Westbrook_states.t, mech_1S_CH4_Westbrook_states('CH4').X, color="g", label="1S_CH4_Westbrook")
        axs[0, 1].plot(mech_2S_CH4_Westbrook_states.t, mech_2S_CH4_Westbrook_states('CH4').X, color="b", label="2S_CH4_Westbrook")
        axs[0, 1].set_ylim([0, 0.1])
        axs[0, 1].set_xlim([0.0000007, 0.003])
        axs[0, 1].set_xlabel("time [s]")
        axs[0, 1].set_ylabel("[$CH_{4}$]")
        axs[0, 1].set_xscale("log")
        axs[0, 1].grid()

        axs[1, 0].plot(gri_states.t, gri_states('O2').X, color="k", label="Gri-Mech3.0")
        axs[1, 0].plot(mech_1S_CH4_MP1_states.t, mech_1S_CH4_MP1_states('O2').X, color="r", label="1S_CH4_MP1")
        axs[1, 0].plot(mech_1S_CH4_Westbrook_states.t, mech_1S_CH4_Westbrook_states('O2').X, color="g", label="1S_CH4_Westbrook")
        axs[1, 0].plot(mech_2S_CH4_Westbrook_states.t, mech_2S_CH4_Westbrook_states('O2').X, color="b", label="2S_CH4_Westbrook")
        axs[1, 0].set_ylim([0, 0.2])
        axs[1, 0].set_xlim([0.0000007, 0.003])
        axs[1, 0].set_xlabel("time [s]")
        axs[1, 0].set_ylabel("[$O_{2}$]")
        axs[1, 0].set_xscale("log")
        axs[1, 0].grid()

        axs[1, 1].plot(gri_states.t, gri_states('CO2').X, color="k", label="Gri-Mech3.0")
        axs[1, 1].plot(mech_1S_CH4_MP1_states.t, mech_1S_CH4_MP1_states('CO2').X, color="r", label="1S_CH4_MP1")
        axs[1, 1].plot(mech_1S_CH4_Westbrook_states.t, mech_1S_CH4_Westbrook_states('CO2').X, color="g", label="1S_CH4_Westbrook")
        axs[1, 1].plot(mech_2S_CH4_Westbrook_states.t, mech_2S_CH4_Westbrook_states('CO2').X, color="b", label="2S_CH4_Westbrook")
        axs[1, 1].set_ylim([0, 0.1])
        axs[1, 1].set_xlim([0.0000007, 0.003])
        axs[1, 1].set_xlabel("time [s]")
        axs[1, 1].set_ylabel("[$CO_{2}$]")
        axs[1, 1].set_xscale("log")
        axs[1, 1].grid()

        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.49, 0.5, 0.5, 0.5), ncol=4)
        
        plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15)
        # plt.tight_layout()
        plt.savefig("methane_combustion_t_{}_p_{}.png".format(initial_temperature, initial_pressure))
        plt.close()
    

if __name__ == ("__main__"):
    main()
