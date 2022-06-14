# Function originally written by : Dr. Vishal Shrivastava
# Modified by: Sahil Bhola
# Function takes in the parameters for the emmisivity computation and returns..
# the temperature prediction

import numpy as np


class RHT_True:
    # Class to generate reference data (considered truth) for Radiative heat transfer
    def __init__(self, T_inf=5.0, npoints=129, dt=1e-2, n_iter=1000, tol=1e-8, plot=True, alpha=3*np.pi/200, gamma=1, delta=5):
        """Init Funciton"""
        # Temperature of the one-dimensional body
        self.T_inf = T_inf
        self.y = np.linspace(0., 1., npoints)      # Coordinates
        # dy^2 to be used in the second derivative
        self.dy2 = (self.y[1]-self.y[0])**2
        # Initial temperature of the body at the coordinates specified above
        self.T = np.zeros_like(self.y)
        self.dt = dt                                # Time step to be used in the simulation
        # Maximum number of iterations to be run during direct solve
        self.n_iter = n_iter
        # Maximum value of residual at which direct solve can be terminated
        self.tol = tol
        # Boolean flag whether to plot the solution at the end of simulation
        self.plot = plot
        # Emissivity parameters
        self.alpha = alpha
        self.gamma = gamma
        self.delta = delta

    def getEmiss(self, T):
        """Funciton returns the emmisivity"""
        # Function to ascertain the local radiative emissivity, given the temperature
        # return 1e-4 * (1. + 5.*np.sin(3*np.pi*T/200.) + np.exp(0.02*T))
        return 1e-4 * (self.gamma + self.delta*np.sin(self.alpha*T) + np.exp(0.02*T))

    def GaussSeidelUpdate(self, T):
        """Gauss Seidel Update"""
        # Evaluate the residual
        res = np.zeros_like(T, dtype=T.dtype)
        T_copy = T.copy()
        emiss = self.getEmiss(T)
        T[1:-1:2] = 0.5 * (T[0:-2:2]+T[2::2]) + 0.5 * self.dy2 * (emiss[1:-1:2] * (self.T_inf**4 - T[1:-1:2]**4) +
                                                                  0.5 * (self.T_inf - T[1:-1:2]))

        T[2:-1:2] = 0.5 * (T[1:-2:2]+T[3::2]) + 0.5 * self.dy2 * (emiss[2:-1:2] * (self.T_inf**4 - T[2:-1:2]**4) +
                                                                  0.5 * (self.T_inf - T[2:-1:2]))

        return np.linalg.norm(T - T_copy)

    def solve(self):
        """Implicit solve"""
        for iteration in range(self.n_iter):

            # Update the states for this iteration
            res_norm = self.GaussSeidelUpdate(self.T)
            # print("%9d\t%E" % (iteration, res_norm))

            if res_norm < self.tol:
                # call("mkdir -p true_solutions", shell=True)
                # np.save("true_solutions/solution_%d" % self.T_inf, self.T)
                break

        return self.T


def compute_prediction(T_inf, alpha=3*np.pi/200, gamma=1, delta=5):
    """Function computes the model prediciton at a given temperature value"""
    rht = RHT_True(
        T_inf=T_inf,
        n_iter=100000,
        tol=1e-13,
        alpha=alpha,
        gamma=gamma,
        delta=delta
    )
    T_prediciton = rht.solve()
    return T_prediciton


def main():
    alpha = 3*np.pi/200
    gamma = 0.02
    delta = 5
    T_inf = 50

    rht = RHT_True(
        T_inf=T_inf,
        n_iter=100000,
        tol=1e-13,
        alpha=alpha,
        gamma=gamma,
        delta=delta
    )
    rht.solve()


if __name__ == "__main__":
    main()
