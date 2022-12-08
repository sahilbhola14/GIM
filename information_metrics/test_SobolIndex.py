import numpy as np
from SobolIndex import SobolIndex


def comp_sobol_indices():
    """Compute the Sobol indices for the linear gaussian model"""
    num_samples = 100
    # Define the model

    def forward_model(theta):
        return comp_linear_gaussian_prediciton(num_samples, theta)

    sobol_index = SobolIndex(
        forward_model=forward_model,
        prior_mean=np.ones(3).reshape(-1, 1),
        prior_cov=np.eye(3),
        global_num_outer_samples=4800,
        global_num_inner_samples=4800,
        model_noise_cov_scalar=1e-1,
        data_shape=(1, num_samples),
        write_log_file=True,
        save_path="./Output/SobolIndex",
    )

    sobol_index.comp_sobol_indices()


def comp_linear_gaussian_prediciton(num_samples, theta):
    """Compute the prediction of a linear Gaussian model."""
    x = np.linspace(0, 1, num_samples)

    # Unequal contribution of the parameters
    # A = np.vander(x, len(theta), increasing=True)

    # Equal contribution of each parameter
    A = np.ones((x.shape[0], len(theta)))
    return np.dot(A, theta).reshape(1, -1)


def main():
    comp_sobol_indices()


if __name__ == "__main__":
    main()
