import numpy as np
import cantera as ct
import matplotlib.pyplot as plt

plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=25)


def ignition():
    # Parameters
    air = "O2:0.21, N2:0.79"
    moles_oxygen = 2
    moles_fuel = 1

    # Initialize Model
    gas = ct.Solution("gri30.yaml")

    gas.set_equivalence_ratio(phi=1.0, fuel="CH4:1", oxidizer=air)
    # gas()
    # Assign initial concentration
    # gas.X = {'CH4': moles_fuel, 'O2': moles_oxygen, 'N2': 7.52}
    # gas()
    gas.TP = 1500, 101325
    print("Initial Fuel concentration : {}".format(moles_fuel))
    print("Initial Oxygen concentration : {}".format(moles_oxygen))
    breakpoint()

    # Equilibrium case
    # gas.equilibrate("UV")
    # gas()

    # Transient case (zero-dim reactor)
    reactor = ct.IdealGasConstPressureReactor(gas)
    sim = ct.ReactorNet([reactor])

    # sim.rtol=1e-6
    # sim.atol=1e-15

    dt = 5e-6
    t_end = 2*1e-3

    states = ct.SolutionArray(gas, extra=['t'])

    for t in np.arange(0, t_end, dt):
        sim.advance(t)
        states.append(reactor.thermo.state, t=1000*t)

    auto_ignition_time, auto_ignition_temp = compute_auto_ignition_time(
        states.T, states.t, dt)

    fig, axs = plt.subplots(2, 2, figsize=(15, 8))
    axs[0, 0].plot(states.t, states.T, color='red', lw=2)
    axs[0, 0].scatter(auto_ignition_time, auto_ignition_temp,
                      marker='s', c="k", s=30, label="Auto ignition")
    axs[0, 0].set_xlabel("time (ms)")
    axs[0, 0].set_ylabel(r"$\textbf{Temperature (K)}$", labelpad=20)
    axs[0, 0].legend(loc="lower right")

    axs[0, 1].plot(states.t, states('OH').X, color='r', lw=2)
    axs[0, 1].set_xlabel("time (ms)")
    axs[0, 1].set_ylabel(r"$\textbf{[OH]}$", labelpad=20)

    axs[1, 0].plot(states.t, states('H').X, color='r', lw=2)
    axs[1, 0].set_xlabel("time (ms)")
    axs[1, 0].set_ylabel(r"$\textbf{[H]}$", labelpad=20)

    axs[1, 1].plot(states.t, states('CH4').X, color='r', lw=2)
    axs[1, 1].set_xlabel("time (ms)")
    axs[1, 1].set_ylabel(r"$\textbf{[CH}_{\textbf{4}}\textbf{]}$", labelpad=20)

    # # plt.xlabel('time(ms)')
    # # plt.ylabel('Temperature')
    # # plt.legend(loc="lower right")
    # # plt.subplot(2,2,2)
    # # plt.plot(states.t,states('OH').X,color='blue')
    # # plt.xlabel('time(ms)')
    # # plt.ylabel('OH mole fraction')
    # # plt.subplot(2,2,3)
    # # plt.plot(states.t,states('H').X,color='green')
    # # plt.xlabel('time(ms)')
    # # plt.ylabel('H mole fraction')
    # # plt.subplot(2,2,4)
    # # plt.plot(states.t,states('CH4').X,color='orange')
    # # plt.xlabel('time(ms)')
    # # plt.ylabel('CH4 mole fraction', labelpad=10)
    plt.tight_layout()
    plt.savefig("methane_combustion.png")
    plt.show()


def compute_auto_ignition_time(temperature, time, dt):
    derivative = np.zeros(temperature.shape[0])
    derivative[1:-1] = (temperature[2:]-temperature[0:-2]) / (2*dt)
    derivative[0] = (temperature[1] - temperature[0]) / dt
    derivative[-1] = (temperature[-1] - temperature[-2]) / dt
    arg_max = np.argmax(derivative)
    return time[arg_max], temperature[arg_max]


def main():
    ignition()


if __name__ == "__main__":
    main()
