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


class mcmc():
    def __init__(self,
                 initial_sample,
                 target_log_pdf_evaluator,
                 num_samples,
                 prop_log_pdf_evaluator,
                 prop_log_pdf_sampler):

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
        return self.accepted_samples/self.num_samples

    def compute_acceptance_probability(self,
                                       current_sample,
                                       proposed_sample,
                                       target_log_pdf_at_current,
                                       target_log_pdf_at_proposed):
        """Compute the acceptance probability"""
        prop_reverse = self.evaluate_proposal_log_pdf(proposed_sample, current_sample)
        prop_forward = self.evaluate_proposal_log_pdf(current_sample, proposed_sample)
        check = target_log_pdf_at_proposed + prop_reverse - target_log_pdf_at_current - prop_forward
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
            proposed_sample = self.sample_proposal_pdf(self.samples[ii-1, :])

            # Evaluate the target log pdf at the proposal sample
            target_log_pdf_at_proposed = self.evaluate_target_log_pdf(proposed_sample)

            # Compute the acceptance probaility
            acceptance_propability = self.compute_acceptance_probability(self.samples[ii-1, :],
                                                                         proposed_sample,
                                                                         target_log_pdf_at_current,
                                                                         target_log_pdf_at_proposed)

            # Accept or reject the sample
            if np.random.rand() < acceptance_propability:
                self.samples[ii, :] = proposed_sample
                self.accepted_samples += 1
            else:
                self.samples[ii, :] = self.samples[ii-1, :]

            if rank == 0 and verbose:
                acceptance_propability = self.compute_acceptance_ratio()
                print("Iteration: ", ii, "Acceptance probability: ", acceptance_propability, flush=True)

    def sample_proposal_pdf(self, current_sample):
        """Sample from the proposal pdf"""
        proposed_sample = self.prop_log_pdf_sampler(current_sample)
        return proposed_sample

    def evaluate_proposal_log_pdf(self, current_sample, proposed_sample):
        proposal_log_pdf = self.prop_log_pdf_evaluator(current_sample, proposed_sample)
        return proposal_log_pdf

class metropolis_hastings(mcmc):
    def __init__(self, initial_sample, target_log_pdf_evaluator, num_samples, sd = None, cov=None):
        dim = len(initial_sample)
        if cov is None:
            if sd is None:
                proposal_cov = ((2.4**2) / dim)*np.eye(dim)
            else:
                proposal_cov = sd*np.eye(dim)
        else:
            proposal_cov = cov

        prop_log_pdf_evaluator  = lambda x, y: self.evaluate_random_walk_pdf(x, y, cov=proposal_cov)
        prop_log_pdf_sampler = lambda x: self.sample_random_walk_pdf(x, cov=proposal_cov)

        # Initialize the parent class
        super().__init__(initial_sample, target_log_pdf_evaluator, num_samples, prop_log_pdf_evaluator, prop_log_pdf_sampler)

    def evaluate_random_walk_pdf(self, x, y, cov):
        """evaluate the random walk pdf"""
        proposal_log_pdf = evaluate_gaussian_log_pdf(x[:, None], y[:, None], cov)
        return proposal_log_pdf.item()

    def sample_random_walk_pdf(self, x, cov):
        """Sample from the random walk pdf"""
        proposed_sample = sample_gaussian(x.reshape(-1, 1), cov, 1)
        return proposed_sample.reshape(-1)

class adaptive_metropolis_hastings(mcmc):
    def __init__(self, initial_sample, target_log_pdf_evaluator, num_samples, adapt_sample_threshold = 100, sd = None, initial_cov=None):

        # Assert statements
        assert adapt_sample_threshold < num_samples, "The adaptation sample threshold must be less than the total number of samples"

        # Initialize the child class

        dim = len(initial_sample)
        self.eps = 1e-8
        if initial_cov is None:
            if sd is None:
                self.sd = 2.4/np.sqrt(dim)
                self.proposal_cov = (self.sd)*np.eye(dim)
            else:
                self.sd = sd
                self.proposal_cov=  self.sd*np.eye(dim)
        else:
            self.proposal_cov = initial_cov

        assert np.all(np.linalg.eigvals(self.proposal_cov) > 0), "The initial covariance matrix must be positive definite"
        assert(self.proposal_cov.shape == (dim, dim)), "The initial covariance matrix must be a square matrix"

        self.adapt_sample_threshold = adapt_sample_threshold
        self.sample_count = 0
        self.k_sample_mean = initial_sample
        self.k_sample_cov = self.proposal_cov

        prop_log_pdf_evaluator  = lambda x, y: self.evaluate_adaptive_random_walk_pdf(x, y)
        prop_log_pdf_sampler = lambda x: self.sample_adaptive_random_walk_pdf(x)


        # Initialize the parent class
        super().__init__(initial_sample, target_log_pdf_evaluator, num_samples, prop_log_pdf_evaluator, prop_log_pdf_sampler)

    def evaluate_adaptive_random_walk_pdf(self, x, y):
        """Evaluate the adaptive random walk pdf"""
        proposal_log_pdf = evaluate_gaussian_log_pdf(x[:, None], y[:, None], self.proposal_cov)
        return proposal_log_pdf.item()

    def sample_adaptive_random_walk_pdf(self, x):
        """Sample from the adaptive random walk pdf"""

        # Update the sample count
        self.increase_count()

        # Update the k-sample statistics
        self.compute_k_sample_statistics(x)

        # Update the proposal covariance
        if self.sample_count > self.adapt_sample_threshold:
            self.update_proposal_covariance()

        proposed_sample = sample_gaussian(x.reshape(-1, 1), self.proposal_cov, 1)
        return proposed_sample.reshape(-1)

    def update_proposal_covariance(self):
        """Function updates the covariance matrix via the samples"""
        # print("Updating the proposal covariance matrix", flush=True)
        # time.sleep(1)
        self.proposal_cov = self.sd*(self.k_sample_cov + self.eps*np.eye(self.k_sample_cov.shape[0]))
        assert np.all(np.linalg.eigvals(self.proposal_cov) > 0), "The proposal covariance matrix is not positive definite"

    def increase_count(self):
        """Function increases the sample count"""
        self.sample_count += 1

    def compute_k_sample_mean(self, x):
        """Function computes the sample mean"""
        self.k_sample_mean = (x / self.sample_count) + ((self.sample_count - 1) / self.sample_count)*self.k_sample_mean
        assert  self.k_sample_mean.shape == x.shape, "The sample mean shape is not correct"

    def compute_k_sample_cov(self, x):
        """Function computes the sample covariance"""

        term_1 = ((self.sample_count - 2) / (self.sample_count - 1))*self.k_sample_cov

        term_2 = np.outer(self.k_sample_mean, self.k_sample_mean)

        term_3 = (1 / self.sample_count)*np.outer(x, x)

        updated_mean = (x / self.sample_count) + ((self.sample_count - 1) / self.sample_count)*self.k_sample_mean

        term_4 = ((self.sample_count)/(self.sample_count - 1))*np.outer(updated_mean, updated_mean)

        self.k_sample_cov = term_1 + term_2 + term_3 - term_4

        assert self.k_sample_cov.shape == (x.shape[0], x.shape[0]), "The sample covariance shape is not correct"

    def compute_k_sample_statistics(self, x):
        """Function computes the k-sample statistics"""
        if self.sample_count > 1:
            self.compute_k_sample_cov(x)

        self.compute_k_sample_mean(x)

def log_normal_log_pdf(sample, mean, cov):
    """Function evaluates the log normal pdf"""
    error = sample - mean
    inner_solve = np.linalg.solve(cov, error)
    inner_exponential_term = error.T@inner_solve
    return -0.5*inner_exponential_term

def main():
    gauss_mean = np.array([1, 0]).reshape(-1, 1)
    gauss_cov = np.ones((2, 2))
    gauss_cov[0, 1] = 0.9
    gauss_cov[1, 0] = 0.9
    true_samples = sample_gaussian(gauss_mean, gauss_cov, 10000)

    target_log_pdf = lambda x: log_normal_log_pdf(x.reshape(-1, 1), gauss_mean, gauss_cov)
    num_samples = 10000
    initial_sample = np.random.randn(2)

    mcmc_sampler = adaptive_metropolis_hastings(
            initial_sample=initial_sample,
            target_log_pdf_evaluator=target_log_pdf,
            num_samples=num_samples,
            adapt_sample_threshold=1000,
            sd=1
            )

    #Compute the samples
    mcmc_sampler.compute_mcmc_samples(verbose=True)

    #Compute acceptance rate
    ar = mcmc_sampler.compute_acceptance_ratio()
    print("Acceptance rate: ", ar)

    # Compute the burn in samples
    burned_samples = sub_sample_data(mcmc_sampler.samples, frac_burn=0.5, frac_use=0.7)

    # Plot the samples
    fig, axs = plot_chains(burned_samples, title="Adaptive Metropolis Hastings")
    plt.show()

if __name__ == "__main__":
    main()
