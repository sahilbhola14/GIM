"""
Developer: Sahil Bhola
Date: 11-17-2022
Description: This script implements the Metropolis-Hastings algorithm
in an objective oriented manner.
"""
import numpy as np
from mpi4py import MPI
from mcmc_utils import *
import time
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class mcmc:
    def __init__(
        self,
        initial_sample,
        target_log_pdf_evaluator,
        num_samples,
        prop_log_pdf_evaluator,
        prop_log_pdf_sampler,
    ):

        self.initial_sample = initial_sample
        self.target_log_pdf_evaluator = target_log_pdf_evaluator
        self.prop_log_pdf_evaluator = prop_log_pdf_evaluator
        self.prop_log_pdf_sampler = prop_log_pdf_sampler
        self.num_samples = num_samples

        self.dim = len(initial_sample)

        self.samples = np.zeros((num_samples, self.dim))
        self.samples[0, :] = self.initial_sample

        self.accepted_samples = 0

    def compute_acceptance_ratio(self):
        """Compute the acceptance ratio"""
        return self.accepted_samples / self.num_samples

    def compute_acceptance_probability(
        self,
        current_sample,
        proposed_sample,
        target_log_pdf_at_current,
        target_log_pdf_at_proposed,
    ):
        """Compute the acceptance probability"""
        prop_reverse = self.evaluate_proposal_log_pdf(proposed_sample, current_sample)
        prop_forward = self.evaluate_proposal_log_pdf(current_sample, proposed_sample)
        check = (
            target_log_pdf_at_proposed
            + prop_reverse
            - target_log_pdf_at_current
            - prop_forward
        )
        if check < 0:
            return np.exp(check)
        else:
            return 1

    def evaluate_target_log_pdf(self, sample):
        """Evaluate the target log pdf"""
        target_log_pdf = self.target_log_pdf_evaluator(sample)
        return target_log_pdf.item()

    def compute_mcmc_samples(self, verbose=False):
        """Compute the samples"""
        # Evaluate the target log pdf at the initial sample
        target_log_pdf_at_current = self.evaluate_target_log_pdf(self.initial_sample)

        for ii in range(1, self.num_samples):

            # Sample from the proposal pdf
            proposed_sample = self.sample_proposal_pdf(self.samples[ii - 1, :])

            # Evaluate the target log pdf at the proposal sample
            target_log_pdf_at_proposed = self.evaluate_target_log_pdf(proposed_sample)

            # Compute the acceptance probaility
            acceptance_propability = self.compute_acceptance_probability(
                self.samples[ii - 1, :],
                proposed_sample,
                target_log_pdf_at_current,
                target_log_pdf_at_proposed,
            )

            # Accept or reject the sample
            if np.random.rand() < acceptance_propability:
                self.samples[ii, :] = proposed_sample
                self.accepted_samples += 1
            else:
                self.samples[ii, :] = self.samples[ii - 1, :]

            if rank == 0 and verbose:
                acceptance_propability = self.compute_acceptance_ratio()
                print(
                    "Iteration: ",
                    ii,
                    "Acceptance probability: ",
                    acceptance_propability,
                    flush=True,
                )

    def sample_proposal_pdf(self, current_sample):
        """Sample from the proposal pdf"""
        proposed_sample = self.prop_log_pdf_sampler(current_sample)
        return proposed_sample

    def evaluate_proposal_log_pdf(self, current_sample, proposed_sample):
        proposal_log_pdf = self.prop_log_pdf_evaluator(current_sample, proposed_sample)
        return proposal_log_pdf


class metropolis_hastings(mcmc):
    def __init__(
        self,
        initial_sample,
        target_log_pdf_evaluator,
        num_samples,
        sd=None,
        initial_cov=None,
    ):

        dim = len(initial_sample)
        if sd is None:
            self.sd = 2.4 / np.sqrt(dim)
        else:
            self.sd = sd

        if initial_cov is None:
            self.proposal_cov = (self.sd) * np.eye(dim)
        else:
            self.proposal_cov = initial_cov

        assert np.all(
            np.linalg.eigvals(self.proposal_cov) > 0
        ), "The covariance matrix is not positive definite"

        prop_log_pdf_evaluator = lambda x, y: self.evaluate_random_walk_pdf(x, y)
        prop_log_pdf_sampler = lambda x: self.sample_random_walk_pdf(x)

        # Initialize the parent class
        super().__init__(
            initial_sample,
            target_log_pdf_evaluator,
            num_samples,
            prop_log_pdf_evaluator,
            prop_log_pdf_sampler,
        )

    def evaluate_random_walk_pdf(self, x, y):
        """evaluate the random walk pdf"""
        proposal_log_pdf = evaluate_gaussian_log_pdf(
            x[:, None], y[:, None], self.proposal_cov
        )
        return proposal_log_pdf.item()

    def sample_random_walk_pdf(self, x):
        """Sample from the random walk pdf"""
        proposed_sample = sample_gaussian(x.reshape(-1, 1), self.proposal_cov, 1)
        return proposed_sample.reshape(-1)


class adaptive_metropolis_hastings(mcmc):
    def __init__(
        self,
        initial_sample,
        target_log_pdf_evaluator,
        num_samples,
        adapt_sample_threshold=100,
        sd=None,
        initial_cov=None,
        eps=1e-8,
        reset_frequency=100,
    ):

        # Assert statements
        assert (
            adapt_sample_threshold < num_samples
        ), "The adaptation sample threshold must be less than the total number of samples"

        # Initialize the child class

        dim = len(initial_sample)
        self.eps = eps
        if sd is None:
            self.sd = 2.4 / np.sqrt(dim)
        else:
            self.sd = sd

        if initial_cov is None:
            self.proposal_cov = (self.sd) * np.eye(dim)
        else:
            self.proposal_cov = initial_cov

        assert np.all(
            np.linalg.eigvals(self.proposal_cov) > 0
        ), "The initial covariance matrix must be positive definite"
        assert self.proposal_cov.shape == (
            dim,
            dim,
        ), "The initial covariance matrix must be a square matrix"

        self.adapt_sample_threshold = adapt_sample_threshold

        self.initial_cov = self.proposal_cov

        self.sample_count = 0
        self.k = 0

        self.old_mean = initial_sample
        self.new_mean = initial_sample

        self.old_cov = self.proposal_cov
        self.new_cov = np.nan * np.ones((dim, dim))

        self.reset_frequency = reset_frequency

        prop_log_pdf_evaluator = lambda x, y: self.evaluate_adaptive_random_walk_pdf(
            x, y
        )
        prop_log_pdf_sampler = lambda x: self.sample_adaptive_random_walk_pdf(x)

        # Initialize the parent class
        super().__init__(
            initial_sample,
            target_log_pdf_evaluator,
            num_samples,
            prop_log_pdf_evaluator,
            prop_log_pdf_sampler,
        )

    def evaluate_adaptive_random_walk_pdf(self, x, y):
        """Evaluate the adaptive random walk pdf"""
        proposal_log_pdf = evaluate_gaussian_log_pdf(
            x[:, None], y[:, None], self.proposal_cov
        )
        return proposal_log_pdf.item()

    def sample_adaptive_random_walk_pdf(self, x):
        """Sample from the adaptive random walk pdf"""

        assert (
            np.prod(x.shape) == self.dim
        ), "The current sample must be a vector of length dim"

        # Update K
        self.k = self.sample_count

        # Update the sample count
        self.increase_count()

        # Update the old mean
        self.update_old_mean()

        # Recursively update the new mean
        self.update_recursive_mean(x)

        # Recursively update the new covariance
        self.update_recursive_cov(x)

        if self.sample_count > self.adapt_sample_threshold:

            # Update the old covariance
            self.update_old_cov()

            # Update the proposal covariance
            self.update_proposal_covariance()

        proposed_sample = sample_gaussian(x.reshape(-1, 1), self.proposal_cov, 1)
        return proposed_sample.reshape(-1)

    def update_proposal_covariance(self):
        """Function updates the covariance matrix via the samples"""
        self.proposal_cov = self.new_cov

    def increase_count(self):
        """Function increases the sample count"""
        self.sample_count += 1

    def update_recursive_mean(self, x):
        """Function updates the recursive mean"""
        # Update new mean
        self.new_mean = (1 / (self.k + 1)) * x + (self.k / (self.k + 1)) * self.old_mean

        assert self.new_mean.shape == x.shape, "The new mean shape is not correct"

    def update_recursive_cov(self, x):
        """Function updates the recursive covariance"""

        multiplier = (
            (self.eps * np.eye(self.dim))
            + (self.k * np.outer(self.old_mean, self.old_mean))
            - (self.k + 1) * np.outer(self.new_mean, self.new_mean)
            + (np.outer(x, x))
        )

        if self.sample_count > self.adapt_sample_threshold:

            if self.sample_count % self.reset_frequency == 0:
                self.reset_mean()
                self.reset_cov()

            # Update new covariance matrix
            self.new_cov = ((self.k - 1) / self.k) * self.old_cov + (
                self.sd / self.k
            ) * multiplier

            assert self.new_cov.shape == (
                x.shape[0],
                x.shape[0],
            ), "The new covariance shape is not correct"
            assert np.all(
                np.linalg.eigvals(self.new_cov) > 0
            ), "The new covariance matrix is not positive definite"

        elif self.sample_count > 1:

            # Update the old covariance matrix
            self.old_cov = ((self.k - 1) / self.k) * self.proposal_cov + (
                self.sd / self.k
            ) * multiplier

        else:

            pass

    def update_old_mean(self):
        """Function updates the old mean"""
        self.old_mean = self.new_mean

    def update_old_cov(self):
        """Function updates the old covariance"""
        self.old_cov = self.new_cov

    def reset_mean(self):
        """Function resets the mean"""
        self.old_mean = self.initial_sample
        self.new_mean = self.initial_sample

    def reset_cov(self):
        """Function resets the covariance"""
        self.old_cov = self.initial_cov
        self.new_cov = np.nan * np.ones((self.dim, self.dim))


def test_gaussian(sample):
    """Function evaluates the log normal pdf
    :param sample: The sample to evaluate (D, 1)
    :return: The log pdf value
    """
    mean = np.arange(1, 3).reshape(-1, 1)
    cov = build_cov_mat(1.0, 1.0, 0.5)  # std, std, correlation
    error = sample - mean
    inner_solve = np.linalg.solve(cov, error)
    inner_exponential_term = error.T @ inner_solve
    return -0.5 * inner_exponential_term


def test_banana(sample):
    """Function evaluates the log normal pdf
    :param sample: The sample to evaluate (D, 1)
    :return: The log pdf value
    """
    a = 1.0
    b = 100.0
    x = sample[0, 0]
    y = sample[1, 0]
    logpdf = (a - x) ** 2 + b * (y - x**2) ** 2
    return -logpdf


def main():
    target_log_pdf = lambda x: test_gaussian(x.reshape(-1, 1))
    # target_log_pdf = lambda x: test_banana(x.reshape(-1, 1))
    num_samples = 10000
    initial_sample = np.random.randn(2)

    mcmc_sampler = adaptive_metropolis_hastings(
        initial_sample=initial_sample,
        target_log_pdf_evaluator=target_log_pdf,
        num_samples=num_samples,
        adapt_sample_threshold=1000,
        reset_frequency=50,
    )

    # Compute the samples
    mcmc_sampler.compute_mcmc_samples(verbose=True)
    print(mcmc_sampler.proposal_cov)

    # Compute acceptance rate
    ar = mcmc_sampler.compute_acceptance_ratio()
    print("Acceptance rate: ", ar)

    # Compute the burn in samples
    burned_samples = sub_sample_data(mcmc_sampler.samples, frac_burn=0.5, frac_use=0.7)

    # Plot the samples
    fig, axs = plot_chains(burned_samples, title="Adaptive Metropolis Hastings")
    plt.show()

    # Plot samples from posterior
    fig, axs, gs = scatter_matrix(
        [burned_samples], labels=[r"$x_1$", r"$x_2$"], hist_plot=False, gamma=0.4
    )

    fig.set_size_inches(7, 7)
    plt.show()


if __name__ == "__main__":
    main()
