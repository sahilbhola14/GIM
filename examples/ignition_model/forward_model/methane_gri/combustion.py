import cantera as ct
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

# Begin user input
initial_temp = 300
initial_pressure = 101325
# End user input


plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=25)
plt.rc("lines", linewidth=2)
plt.rc("axes", labelpad=30, titlepad=20)

air = "O2: 0.21, N2: 0.79"
phi = np.linspace(0.25, 4, 100)
temp = np.zeros_like(phi)

# Adiabatic flame temp
gas = ct.Solution("gri30.yaml")
for ii, iratio in enumerate(phi):
    gas.set_equivalence_ratio(phi=iratio, fuel="CH4:1", oxidizer=air)
    gas.TP=initial_temp, initial_pressure
    gas.equilibrate("UV")
    temp[ii] = gas.T

output = np.zeros((phi.shape[0], 2))
output[:, 0] = phi
output[:, 1] = temp
np.savetxt("methane_gri_data.dat", output, fmt="%10.5f", delimiter=' ', header='Initial_temperature : %f,  Initial_pressure : %f \n Parameters : Equivalence_ratio, Adiabatic_temperature'%(initial_temp, initial_pressure),)
# np.save("methane_gri_adiabatic_temperature_vs_equivalence_ratio", temp)
# fig, axs = plt.subplots(figsize=(10, 8))
# axs.scatter(phi, temp, lw=2, c='r')
# axs.set_xlabel(r"$\phi$")
# axs.set_ylabel(r"Adiabatic temperature (K)")
# axs.set_title(r"Adiabatic temperature v.s. $\phi$")
# axs.xaxis.set_minor_locator(MultipleLocator(0.25))
# axs.set_ylim([500, 3000])
# plt.tight_layout()
# plt.savefig("adiabatic_temperature_vs_equivalence_ratio.png")
# plt.show()
