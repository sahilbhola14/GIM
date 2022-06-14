"""Original code by Vishal Srivastava"""
import numpy as np


def setJacobian(jacT, jacbeta, T, T_inf, beta, dy2, relay=False):
    jacT[0, 0] = -1.0
    jacT[-1, -1] = -1.0

    ii = np.linspace(1, jacT.shape[0]-2, jacT.shape[0]-2).astype(int)
    jacT[ii, ii-1] = 1. / dy2
    jacT[ii, ii] = - 2. / dy2 - 2E-3 * beta[ii] * T[ii]**3
    jacT[ii, ii+1] = 1. / dy2

    jacbeta[ii, ii] = 5E-4 * (T_inf**4 - T[ii]**4)

    if relay:
        return jacT, jacbeta


class rht:
    """radiative heat transfer class"""

    def __init__(self, T_inf=5.0, n_spatial=129, dt=1e-2, n_iter=1000, tol=1e-8, net=None):
        self.n_spatial = n_spatial
        # Temperature of the one-dimensional body
        self.T_inf = T_inf
        self.y = np.linspace(0., 1., self.n_spatial)      # Coordinates
        # dy^2 to be used in the second derivative
        self.dy2 = (self.y[1]-self.y[0])**2
        # Initial temperature of the body at the coordinates specified above
        self.T = np.zeros_like(self.y)

        # Augmentation profiles of the model
        self.beta = np.ones_like(self.y)
        self.dt = dt                                # Time step to be used in the simulation
        # Maximum number of iterations to be run during direct solve
        self.n_iter = n_iter
        self.print_iter = int(self.n_iter / 50)
        # Maximum value of residual at which direct solve can be terminated
        self.tol = tol
        self.res = np.zeros_like(self.y)             # Residuals
        # \partial R / \partial T
        self.jacT = np.zeros((self.n_spatial, self.n_spatial))
        # \partial R / \partial beta
        self.jacbeta = np.zeros((self.n_spatial, self.n_spatial))
        self.net = net  # Network parameterization of beta

    def getFeatures(self):
        """Function computes the required featured for the NN
        outputs:
        features : (2, n_spatial) : features for the NN
        """
        features = np.zeros((2, self.n_spatial))
        features[0, :] = self.T_inf / self.T_inf
        features[1, :] = self.T / self.T_inf

        return features

    def evalResidual(self):
        """Funcition computes the residual for the ODE solve"""
        self.res[1:-1] = (self.T[0:-2]-2*self.T[1:-1]+self.T[2:]) / \
            self.dy2 + 5E-4*self.beta[1:-1]*(self.T_inf**4-self.T[1:-1]**4)
        self.res[0] = -self.T[0]
        self.res[-1] = -self.T[-1]
        return self.res

    def compute_beta(self, features):
        """Function computes the beta field predicion"""
        # Assign the network inputs
        self.net.assign_network_IO(network_input=features)
        # Deploy Network
        self.net.deploy_network()
        # Collect prediction
        self.beta = self.net.prediction.ravel()

    def implicitEulerUpdate(self):
        """Funciton updates the states using implicit Euler time stepping"""
        features = self.getFeatures()
        # deploying the network for computing beta
        if self.net != None:
            self.compute_beta(features)
        setJacobian(self.jacT, self.jacbeta, self.T,
                    self.T_inf, self.beta, self.dy2)

        self.evalResidual()
        self.T = self.T + \
            np.linalg.solve(
                np.eye(self.y.shape[0])/self.dt - self.jacT, self.res)
        return np.linalg.norm(self.res)

    def direct_solve(self):
        """Funciton solves the forward model"""
        for iteration in range(self.n_iter):
            res_norm = self.implicitEulerUpdate()
            # Check if the residual is within tolerance, if yes, save the data and exit the simulation, if no, continue
            if res_norm < self.tol:
                # print("Forward model converged at residual : {0:.2e} (T_inf : {1:.1f})".format(res_norm, self.T_inf))
                break
            elif iteration == self.n_iter-1:
                print(
                    "Maximum iterations reached! residual norm : {0:.5f}".format(res_norm))
        return self.res

    def sub_sample(self, sample_idx):
        """Function sub-samples the prediction"""
        return self.T[sample_idx]

    def adjoint_solve(self, data, model_noise_cov, scaling, sample_idx):
        """Funciton solves the adjoint equations"""
        # Compute the permutation matrix
        P = np.zeros((sample_idx.shape[0], self.n_spatial))
        P[np.arange(sample_idx.shape[0]), sample_idx] = 1

        # computing jac_cost_state
        prediction = self.sub_sample(sample_idx)
        if model_noise_cov == 0:
            jac_cost_state = (scaling*(data - prediction)).reshape(1, -1)
        else:
            jac_cost_state = (scaling*(data - prediction) /
                              model_noise_cov).reshape(1, -1)
        jac_cost_state = jac_cost_state@P

        # computing jac_cost_beta
        jac_cost_beta = np.zeros((1, self.n_spatial))
        # computing jac_res_state
        jac_res_state = self.jacT
        # computing jac_res_beta
        jac_res_beta = self.jacbeta

        # solving the adjoint equation
        psi = np.linalg.solve(jac_res_state.T, jac_cost_state.T)
        nabla_cost_beta = jac_cost_beta - psi.T@jac_res_beta

        return nabla_cost_beta

    def get_jacobians(self):
        """Function returns the jacobians"""

        # computing jac_cost_beta
        jac_cost_beta = np.zeros((1, self.n_spatial))
        # computing jac_res_state
        jac_res_state = self.jacT
        # computing jac_res_beta
        jac_res_beta = self.jacbeta

        jac = (jac_cost_beta, jac_res_state, jac_res_beta)

        return jac

    def infer_beta(self):
        """Function return the beta value"""
        # Compute the features
        features = self.getFeatures()
        # Compute beta
        self.compute_beta(features=features)

        return self.beta
