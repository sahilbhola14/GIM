import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)


class unscented_quadrature():
    def __init__(self, mean, cov, integrand, alpha=1, beta=0, kappa=0):
        self.mean = mean
        self.cov = cov
        self.alpha = 1
        self.beta = beta
        self.kappa = kappa
        self.integrand = integrand

    def compute_quadrature_points(self):
        """Function commputs the unscented quadrature points
        Ouputs:
            quad_points: quadrature points

        """
        # Definitions
        dim = self.cov.shape[0]
        quad_points = np.zeros((dim, 2*dim+1))
        lam = self.alpha*self.alpha*(dim+self.kappa) - dim

        L = np.linalg.cholesky(self.cov+np.eye(dim)*1e-8)

        # Quadrature points
        quad_points[:, 0] = self.mean.ravel()
        for ipoint in range(1, dim+1):
            quad_points[:, ipoint] = self.mean.ravel(
            ) + np.sqrt(dim+lam)*L[:, ipoint-1]
            quad_points[:, ipoint+dim] = self.mean.ravel() - np.sqrt(dim +
                                                                     lam)*L[:, ipoint-1]

        return quad_points

    def compute_weights(self):
        """Function computs the weights"""
        dim = self.cov.shape[0]
        lam = self.alpha*self.alpha*(dim+self.kappa) - dim

        W0m = lam / (dim + lam)
        W0c = lam / (dim + lam) + (1-self.alpha*self.alpha + self.beta)
        Wim = 1 / (2*(dim + lam))
        Wic = 1 / (2*(dim + lam))
        return (W0m, Wim, W0c, Wic)

    def compute_integeral(self):
        """Function comptues the integral"""
        # Compute the quadrature points
        quadrature_points = self.compute_quadrature_points()
        # Compute the weights
        weights = self.compute_weights()
        # Compute integrand at the quadrature points
        model_prediction = self.integrand(quadrature_points)

        integral_mean = weights[0]*model_prediction[:, 0]
        integral_mean += np.sum(weights[1]*model_prediction[:, 1:], axis=1)

        def compute_error(x): return (x-integral_mean).reshape(-1, 1)

        def compute_outer(x):
            error = compute_error(x)
            return error@error.T

        integral_cov = weights[2]*compute_outer(model_prediction[:, 0])
        for ii in range(1, model_prediction.shape[1]):
            integral_cov += weights[-1]*compute_outer(model_prediction[:, ii])

        return integral_mean, integral_cov


class gauss_hermite_quadrature():
    def __init__(self, mean, cov, integrand, num_points):
        self.mean = mean
        self.cov = cov
        self.integrand = integrand
        self.num_points = num_points

    def compute_quadrature_points_and_weights(self):
        """Function comptues the quadrature points (unrotated) and weights using the Golub-Welsch Algorithm
        Adapted from : Dr. Alex gorodedsky notes on Inference, Estimation, and Learning
        NOTE: Uses hermite polynomials"""
        alpha = np.zeros(self.num_points)
        beta = np.sqrt(np.arange(1, self.num_points))
        diagonals = [alpha, beta, beta]
        J = diags(diagonals, [0, 1, -1]).toarray()
        quad_points, evec = np.linalg.eig(J)
        weights = evec[0, :]**2

        unrotated_points, weights = self.tensorize_quadrature_points_and_weights(
            quad_points=quad_points,
            weights=weights
        )
        quad_points = self.rotate_points(unrotated_points=unrotated_points)

        return quad_points, weights

    def tensorize(self, vector):
        """Function tensorizes an input vector"""
        n1d = vector.shape[0]
        twodnodes = np.zeros((n1d*n1d, 2))
        ind = 0
        for ii in range(n1d):
            for jj in range(n1d):
                twodnodes[ind, :] = np.array([vector[ii], vector[jj]])
                ind += 1
        return twodnodes

    def tensorize_quadrature_points_and_weights(self, quad_points, weights):
        """Tensorize the quadrature points and weights, essentially creating a meshgrid"""
        d = self.mean.shape[0]
        if d == 1:
            quad_points_T = quad_points.reshape(1, -1)
            weights_T = weights
        else:
            assert(d == 2), "Tensorization net implelented for more than 2 dimensions"
            quad_points_T = self.tensorize(quad_points).T
            weights_T = self.tensorize(weights)
            weights_T = np.prod(weights_T, axis=1)

        return quad_points_T, weights_T

    def rotate_points(self, unrotated_points):
        """Function rotates the points"""
        d = self.mean.shape[0]
        L = np.linalg.cholesky(self.cov+np.eye(d)*1e-8)
        rotated_points = np.zeros(unrotated_points.shape)
        for ipoint in range(rotated_points.shape[1]):
            rotated_points[:, ipoint] = self.mean.ravel() + np.dot(L, unrotated_points[:, ipoint])
        return rotated_points

    def compute_integeral(self):
        """Function comptues the intergral"""
        # Compute the quadrature points and weights
        quadrature_points, weights = self.compute_quadrature_points_and_weights()
        # Compute integrand at the quadrature points
        model_prediction = self.integrand(quadrature_points)
        # Compute the integral
        integral = np.array([model_prediction[:, ii]*weights[ii] for ii in range(weights.shape[0])])
        return np.sum(integral, axis=0)


def test_function(x):
    """test function to test the quadrature rules"""
    num_samples = x.shape[1]
    output = np.zeros((2, num_samples))
    output[0, :] = x[0, :]*np.tanh(x[0, :]*x[1, :])
    output[1, :] = np.sqrt(x[1, :]**2)
    return output


def sample_gaussian(mean, cov, num_samples):
    """Function samples the gaussian"""
    d = cov.shape[0]
    L = np.linalg.cholesky(cov+np.eye(d)*1e-8)
    noise = L@np.random.randn(d*num_samples).reshape(d, num_samples)
    return mean+noise


def plot_points(num_samples, gaussian_mean, gaussian_cov, integrand_function, view_points=None):
    """Function plots the points"""
    gaussian_samples = sample_gaussian(
        mean=gaussian_mean,
        cov=gaussian_cov,
        num_samples=num_samples
    )

    if view_points is not None:
        model_evaluated_at_view_points = integrand_function(x=view_points)

    # Compute the model prediction
    model_samples = integrand_function(x=gaussian_samples)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].scatter(gaussian_samples[0, :],
                   gaussian_samples[1, :], color='red', s=5)
    axs[1].scatter(model_samples[0, :], model_samples[1, :], color='red', s=5)
    if view_points is not None:
        axs[0].scatter(view_points[0, :], view_points[1, :], color='k', s=60)
        axs[1].scatter(model_evaluated_at_view_points[0, :],
                       model_evaluated_at_view_points[1, :], color='k', s=60)

    axs[0].set_title(r"Input space")
    axs[0].set_xlabel(r"$X_1$")
    axs[0].set_ylabel(r"$X_2$")
    axs[1].set_title(r"Output space")
    axs[1].set_xlabel(r"$Y_1$")
    axs[1].set_ylabel(r"$Y_2$")
    plt.tight_layout()
    plt.show()


def reference_cov():
    "test gaussian"
    std1 = 1
    std2 = 2
    rho = 0.9
    cov = np.array([[std1*std1, rho * std1 * std2],
                   [rho * std1 * std2, std2*std2]])
    return cov


def main():
    gaussian_mean = np.zeros((2, 1))
    gaussian_cov = reference_cov()

    unscented_quad = unscented_quadrature(
            mean=gaussian_mean,
            cov=gaussian_cov,
            integrand=test_function,
            )
    mean, cov = unscented_quad.compute_integeral()
    print(mean)

    gh = gauss_hermite_quadrature(
            mean=gaussian_mean,
            cov=gaussian_cov,
            integrand=test_function,
            num_points=3
            )
    gh_int = gh.compute_integeral()
    print(gh_int)
    # gh_points, gh_weights = gh.compute_quadrature_points_and_weights()

    # plot_points(1000, gaussian_mean=gaussian_mean, gaussian_cov=gaussian_cov, integrand_function=test_function, view_points=gh_points)



if __name__ == "__main__":
    main()
