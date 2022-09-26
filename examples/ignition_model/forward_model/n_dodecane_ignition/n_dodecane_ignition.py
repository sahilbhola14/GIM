import numpy as np
import cantera as ct
import matplotlib.pyplot as plt
from generate_custom_gas_input import gen_gas_file
from mpi4py import MPI
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=25)


def n_dodecane_combustion(equivalence_ratio, initial_temp, pre_exponential_parameter=27.38, activation_energy=31944.0, initial_fuel_moles=1, campaign_path=None):
    #  Compute the pre-exponential factor
    pre_exponential_factor = compute_pre_exponential_factor(
        equivalence_ratio=equivalence_ratio,
        initial_temp=initial_temp,
        pre_exponential_parameter=pre_exponential_parameter
    )
    # Update the gas file
    gen_gas_file(
        reaction_1_pre_exp_factor=float(pre_exponential_factor),
        reaction_1_activation_energy=float(activation_energy),
        campaign_path=campaign_path
    )

    # Initial_conc
    moles_fuel = initial_fuel_moles
    moles_oxygen = moles_fuel/(equivalence_ratio*(1/18.5))

    # Initialize Model
    if campaign_path is not None:
        file_path = os.path.join(campaign_path, "custom_gas_rank_"+str(rank)+".yaml")
    else:
        file_path = "./custom_gas_rank_"+str(rank)+".yaml"
    gas = ct.Solution(file_path)
    # # Assign initial concentration
    gas.X = {'C12H26': moles_fuel, 'O2': moles_oxygen}
    gas.TP = initial_temp, 101325
    # gas.TP = initial_temp, 6000000

    # Reactor
    t_end = 1*1e-3
    reactor = ct.IdealGasConstPressureReactor(gas)
    sim = ct.ReactorNet([reactor])
    states = ct.SolutionArray(gas, extra=['t'])

    sim.atol = 1e-15
    sim.rtol = 1e-6

    # Adaptive stepping
    # while sim.time < t_end:
    #     print("Time : {0:.5f} [ms]".format(sim.time*1000))
    #     sim.step()
    #     states.append(reactor.thermo.state, t=sim.time)

    # Fixed step size
    dt = 5e-7
    for t in np.arange(0, t_end, dt):
        # print("Time : {0:.5f} [ms]".format(sim.time*1000))
        sim.advance(t)
        states.append(reactor.thermo.state, t=t)

    # Ignition timing
    auto_ignition_time, auto_ignition_temp = compute_auto_ignition_time(
        temperature=states.T,
        time=states.t,
        dt=dt
    )

    # """
    fig, axs = plt.subplots(3, 2, figsize=(15, 8))
    axs[0, 0].plot(states.t, states.T, color='red', lw=2)
    axs[0, 0].scatter(auto_ignition_time, auto_ignition_temp,
                      marker='s', c="k", s=30, label="Auto ignition")
    axs[0, 0].set_xlabel("time (s)")
    axs[0, 0].set_ylabel(r"Temperature [K]", labelpad=20)
    axs[0, 0].legend(loc="lower right")
    axs[0, 0].set_xscale("log")

    axs[0, 1].plot(states.t, states('C12H26').X, color='r', lw=2)
    axs[0, 1].set_xlabel("time (s)")
    axs[0, 1].set_ylabel(r"$C_{12}H_{26}$", labelpad=20)
    axs[0, 1].set_xscale("log")

    axs[1, 0].plot(states.t, states('O2').X, color='r', lw=2)
    axs[1, 0].set_xlabel("time (s)")
    axs[1, 0].set_ylabel(r"$O_{2}$", labelpad=20)
    axs[1, 0].set_xscale("log")

    axs[1, 1].plot(states.t, states('H2O').X, color='r', lw=2)
    axs[1, 1].set_xlabel("time (s)")
    axs[1, 1].set_ylabel(r"$H_{2}O$", labelpad=20)
    axs[1, 1].set_xscale("log")

    axs[2, 0].plot(states.t, states('CO').X, color='r', lw=2)
    axs[2, 0].set_xlabel("time (s)")
    axs[2, 0].set_ylabel(r"$CO$", labelpad=20)
    axs[2, 0].set_xscale("log")

    axs[2, 1].plot(states.t, states('CO2').X, color='r', lw=2)
    axs[2, 1].set_xlabel("time (s)")
    axs[2, 1].set_ylabel(r"$CO_{2}$", labelpad=20)
    axs[2, 1].set_xscale("log")

    plt.tight_layout()
    figure_path = os.path.join(campaign_path, "true_n_dodecane_combustion_T_{}_phi_{}_rank_{}.png".format(initial_temp, equivalence_ratio, rank))
    plt.savefig(figure_path)
    plt.close()
    # """

    return auto_ignition_time


def compute_auto_ignition_time(temperature, time, dt):
    """Function computes the ignition time via derivatives"""
    derivative = np.zeros(temperature.shape[0])
    derivative[1:-1] = (temperature[2:]-temperature[0:-2]) / (2*dt)
    derivative[0] = (temperature[1] - temperature[0]) / dt
    derivative[-1] = (temperature[-1] - temperature[-2]) / dt
    arg_max = np.argmax(derivative)
    return time[arg_max], temperature[arg_max]


# def compute_ignition_time(states, t_end):
#     if states.t[states.T>1500].shape[0] == 0:
#         ignition_time = t_end
#     else:
#         ignition_time = states.t[states.T>1500][0]
#     return ignition_time


def compute_pre_exponential_factor(pre_exponential_parameter, equivalence_ratio=0.5, initial_temp=300):
    """Function computes the pre_exponential_factor"""
    lambda_0 = pre_exponential_parameter
    lambda_1 = -2.13
    lambda_2 = -2.05
    lambda_3 = 1.89
    lambda_4 = -1.0*10**(-2)
    lambda_5 = 2.87*10**(-4)
    lambda_6 = 8.43

    log_pre_exponential_factor = lambda_0 + (lambda_1*np.exp(lambda_2*equivalence_ratio)) + lambda_3*np.tanh(
        ((lambda_4 + (lambda_5*equivalence_ratio))*initial_temp) + lambda_6)

    return np.exp(log_pre_exponential_factor)


def main():
    # Begin user input
    equivalence_ratio = 0.5
    initial_temp = 1000  # K
    # End user input

    n_dodecane_combustion(
        equivalence_ratio=equivalence_ratio,
        initial_temp=initial_temp,
        pre_exponential_parameter=30.0,
        activation_energy=35000.0
    )


if __name__ == "__main__":
    main()
