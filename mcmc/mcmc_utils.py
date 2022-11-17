import numpy as np

def sample_gaussian(mean, cov, n):
    chol = np.linalg.cholesky(cov)
    return mean + np.dot(chol, np.random.randn(mean.shape[0], n))

def evaluate_gaussian_log_pdf(sample, mean, cov):
    """Function evaluates the log normal pdf"""
    error = sample - mean
    inner_solve = np.linalg.solve(cov, error)
    log_pdf = -0.5*np.dot(error.T, inner_solve)
    return log_pdf
