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
    gas = ct.Solution("1S_CH4_MP1.yaml")
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
    gas = ct.Solution("1S_CH4_Westbrook.yaml")
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
    gas = ct.Solution("2S_CH4_Westbrook.yaml")
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

def main():
    # Begin user input
    initial_temperature = 1400 # K
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
