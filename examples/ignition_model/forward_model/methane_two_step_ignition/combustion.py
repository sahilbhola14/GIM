import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=25)
plt.rc("lines", linewidth=2)
plt.rc("axes", labelpad=30, titlepad=20)

class methane_combustion():
    def __init__(self, initial_temp, initial_pressure, default_pre_exp=1.5e+7):
        self.initial_temp = initial_temp
        self.initial_pressure = initial_pressure
        
        # Combustion model parameters
        self.default_pre_exp = default_pre_exp
        
        # Initializations
        self.initial_conc = None
    
    def compute_initial_conc(self, equivalence_ratio):
        """Function computes the initial concentration"""
        moles_methane = 1
        factor = equivalence_ratio*(1/2)
        moles_oxygen = moles_methane/factor
        moles_nitrogen = (0.79/0.21)*moles_oxygen
        initial_conc = {
                "CH4": moles_methane,
                "O2": moles_oxygen,
                "N2": moles_nitrogen
                }
        return initial_conc

    def compute_adiabatic_temperature(self, equivalence_ratio, theta=None):
        """Function comptues the adiabatic temperature"""

        # Load the gas file (internal load)
        if theta is None:
            gas_file = generate_methane_oxygen_gas_file(pre_exp_factor=self.default_pre_exp)
        else:
            parameterized_pre_exp = self.compute_pre_exp_factor(theta, equivalence_ratio)
            gas_file = generate_methane_oxygen_gas_file(pre_exp_factor=parameterized_pre_exp)

        gas = ct.Solution(yaml=gas_file)

        # Update the concentration
        self.initial_conc = self.compute_initial_conc(equivalence_ratio=equivalence_ratio)
        gas.X = {
                "CH4": self.initial_conc["CH4"],
                "O2": self.initial_conc["O2"],
                "N2": self.initial_conc["N2"]
                }

        # Update the initial temperature and pressure
        gas.TP = self.initial_temp, self.initial_pressure
        
        # Forward solve
        gas.equilibrate("UV")

        return gas.T

    def compute_pre_exp_factor(self, theta, equivalence_ratio):
        """Function computes the pre exponential factor
        All thetas' are assumed to be normally distributed, i.e. theta_i ~ N(0, 1)
        """

        lambda_1 = 20 + (2)*theta[0] 
        lambda_2 = theta[1] 
        lambda_3 = theta[2] 
        log_A = lambda_1 + np.tanh((lambda_2 + lambda_3*equivalence_ratio)*(self.initial_temp/1000))

        return np.exp(log_A)

    def compute_log_pre_exp_factor_distribution(self, equivalence_ratio):
        """Function computes the log pre-exponential distribution assuming normal prior"""
        num_samples = 1000
        theta_samples = np.random.randn(num_samples*3).reshape(3, -1)
        output_samples = np.zeros(num_samples)

        for ii in range(num_samples):
            itheta = theta_samples[:, ii]
            output_samples[ii] = np.log(self.compute_pre_exp_factor(theta=itheta, equivalence_ratio=equivalence_ratio))
        
        # print(np.exp(np.mean(output_samples)))
        plt.figure()
        plt.hist(output_samples, density=True, bins=50)
        plt.xlabel(r"$\log(A)$")
        plt.ylabel(r"p($\log(A)$)")
        plt.tight_layout()
        plt.show()


def generate_methane_oxygen_gas_file(pre_exp_factor):
    """Function generates the cutom gas file"""

    custom_gas_data = '''
    units: {length: cm, time: s, quantity: mol, activation-energy: cal/mol}

    phases:
    - name: gas
      thermo: ideal-gas
      elements: [O, H, C, N, Ar]
      species: [CH4, O2, CO, H2O, CO2, N2]
      kinetics: gas
      trasport: mixture-averaged
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
    - equation: CH4 + 1.5 O2 => CO + 2 H2O  # Reaction 1
      rate-constant: {A: %f, b: 0 , Ea: 30000.0}
      orders: {CH4: -0.3, O2: 1.3}
      negative-orders: true
    - equation: CO + 0.5 O2 => CO2 # Reaction 2
      rate-constant: {A: 1.0e+14, b: 0 , Ea: 40000.0}
      orders: {CO: 1, H2O: 0.5, O2: 0.25}
      nonreactant-orders: true
    - equation: CO2 => CO + 0.5 O2 # Reaction 3
      rate-constant: {A: 5.0e+8, b: 0, Ea: 40000.0}
      orders: {CO2: 1}
    '''%(pre_exp_factor)

    return custom_gas_data


def main():
    # Begin user input
    initial_temp = 300
    initial_pressure = 101325
    methane_gri_data_file_path = "./methane_gri_data.dat"
    # End user input

    # Load the gri data
    methane_gri_data = np.loadtxt(methane_gri_data_file_path, skiprows=2)

    phi = methane_gri_data[:, 0]
    adiabatic_temp = np.zeros_like(phi)

    tic = time.time()
    # Initialize the model
    model = methane_combustion(
            initial_temp=initial_temp,
            initial_pressure=initial_pressure,
            )


    # # Compute the adiabatic temperature
    for ii, iphi in enumerate(phi):
        adiabatic_temp[ii] = model.compute_adiabatic_temperature(
                equivalence_ratio=iphi,
                )

    fig, axs = plt.subplots(figsize=(10, 7))
    axs.plot(methane_gri_data[:, 0], methane_gri_data[:, 1], color="k", label="GRI-Mech3.0")
    axs.plot(methane_gri_data[:, 0], adiabatic_temp, color="r", label="2-step Mechanism")
    axs.legend(loc="upper right", framealpha=1)
    axs.set_xlabel(r"Equivalence ratio, $\phi$")
    axs.set_ylabel(r"Temperature [K]")
    axs.set_ylim([1000, 3000])
    axs.set_xlim([0, 4])
    axs.grid()
    plt.tight_layout()
    plt.savefig("adiabatic_temperature.png")
    plt.close()

if __name__ == "__main__":
    main()
