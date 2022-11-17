"""
Developer: Sahil Bhola
Date: 11-17-2022
Description: This script implements the Metropolis-Hastings algorithm
in an objective oriented manner.
"""
import numpy as np
from mcmc_utils import *

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
        pass

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

    def compute_mcmc_samples(self):
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
        """Evaluate the random walk pdf"""
        proposal_log_pdf = evaluate_gaussian_log_pdf(x[:, None], y[:, None], cov)
        return proposal_log_pdf.item()

    def sample_random_walk_pdf(self, x, cov):
        """Sample from the random walk pdf"""
        proposed_sample = sample_gaussian(x.reshape(-1, 1), cov, 1)
        return proposed_sample.reshape(-1)

class adaptive_metropolis_hastings(mcmc):
    def __init__(self, initial_sample, target_log_pdf_evaluator, num_samples):
        prop_log_pdf_evaluator  = lambda x, y: self.evaluate_adaptive_random_walk_pdf(x, y)
        prop_log_pdf_sampler = lambda x: self.sample_adaptive_random_walk_pdf(x)

        # Initialize the parent class
        super().__init__(initial_sample, target_log_pdf_evaluator, num_samples, prop_log_pdf_evaluator, prop_log_pdf_sampler)

    def evaluate_adaptive_random_walk_pdf(self, x, y):
        """Evaluate the adaptive random walk pdf"""
        pass

    def sample_adaptive_random_walk_pdf(self, x):
        """Sample from the adaptive random walk pdf"""
        pass



def log_normal_log_pdf(sample, mean, cov):
    """Function evaluates the log normal pdf"""
    error = sample - mean
    inner_solve = np.linalg.solve(cov, error)
    inner_exponential_term = error.T@inner_solve
    return -0.5*inner_exponential_term

def main():
    gauss_mean = np.array([1, 0]).reshape(-1, 1)
    gauss_cov = np.ones((2, 2))
    gauss_cov[0, 1] = 0.5
    gauss_cov[1, 0] = 0.5

    target_log_pdf = lambda x: log_normal_log_pdf(x.reshape(-1, 1), gauss_mean, gauss_cov)
    num_samples = 100000
    initial_sample = np.random.randn(2)

    mcmc_sampler = metropolis_hastings(
            initial_sample=initial_sample,
            target_log_pdf_evaluator=target_log_pdf,
            num_samples=num_samples
            )

    #Compute the samples
    mcmc_sampler.compute_mcmc_samples()




if __name__ == "__main__":
    main()
