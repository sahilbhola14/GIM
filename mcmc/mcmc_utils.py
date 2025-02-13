import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors

plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=20)
plt.rc("axes", labelsize=20, titlepad=20)
plt.rc("lines", linewidth=3)


def sample_gaussian(mean, cov, n):
    """Function samples from a Gaussian distribution
    :param mean: mean of the Gaussian distribution
    :param cov: covariance matrix of the Gaussian distribution
    :param n: number of samples to draw
    :return: samples from the Gaussian distribution
    """
    assert np.allclose(cov, cov.T), "Covariance matrix must be symmetric"
    assert np.all(
        np.linalg.eigvals(cov) > 0
    ), "Covariance matrix must be positive definite"
    chol = np.linalg.cholesky(cov + 1e-8 * np.eye(cov.shape[0]))
    return mean + np.dot(chol, np.random.randn(mean.shape[0], n))


def evaluate_gaussian_log_pdf(sample, mean, cov):
    """Function evaluates the log normal pdf
    :param sample: sample to evaluate the log pdf at
    :param mean: mean of the Gaussian distribution
    :param cov: covariance matrix of the Gaussian distribution
    :return: log pdf of the sample
    """
    assert np.allclose(cov, cov.T), "Covariance matrix must be symmetric"
    assert np.all(
        np.linalg.eigvals(cov) > 0
    ), "Covariance matrix must be positive definite"
    error = sample - mean
    inner_solve = np.linalg.solve(cov, error)
    log_pdf = -0.5 * np.dot(error.T, inner_solve)
    return log_pdf


def build_cov_mat(std1, std2, rho):
    """Build a covariance matrix for a bivariate Gaussian distribution
    :param std1: standard deviation of the first variable
    :param std2: standard deviation of the second variable
    :param rho: correlation coefficient
    :return: covariance matrix
    """
    assert std1 > 0, "standard deviation must be greater than 0"
    assert std2 > 0, "standard deviation must be greater than 0"
    assert np.abs(rho) <= 1, "correlation must be betwene -1 and 1"
    return np.array([[std1**2, rho * std1 * std2], [rho * std1 * std2, std2**2]])


def sub_sample_data(samples, frac_burn=0.2, frac_use=0.7):
    """Subsample data by burning off the front fraction and using another fraction
    Written by: Alex Gorodedsky
    :param samples: samples to subsample
    :param frac_burn: fraction of samples to burn off
    :param frac_use: fraction of samples to use
    :return: subsampled samples

    """
    num_samples = samples.shape[0]
    inds = np.arange(num_samples, dtype=np.int)
    start = int(frac_burn * num_samples)
    inds = inds[start:]
    num_samples = num_samples - start
    step = int(num_samples / (num_samples * frac_use))
    inds2 = np.arange(0, num_samples, step)
    inds = inds[inds2]
    return samples[inds, :]


def scatter_matrix(
    samples,  # list of chains
    mins=None,
    maxs=None,
    upper_right=None,
    specials=None,
    hist_plot=True,  # if false then only data
    nbins=200,
    gamma=0.5,
    labels=None,
):

    """Scatter matrix hist
    Written by: Alex Gorodedsky
    :param samples: list of samples
    """

    nchains = len(samples)
    dim = samples[0].shape[1]

    if mins is None:
        mins = np.zeros((dim))
        maxs = np.zeros((dim))

        for ii in range(dim):
            # print("ii = ", ii)
            mm = [np.quantile(samp[:, ii], 0.01, axis=0) for samp in samples]
            # print("\t mins = ", mm)
            mins[ii] = np.min(mm)
            mm = [np.quantile(samp[:, ii], 0.99, axis=0) for samp in samples]
            # print("\t maxs = ", mm)
            maxs[ii] = np.max(mm)

            # if specials is not None:
            #     if isinstance(specials, list):
            #         minspec = np.min([spec["vals"][ii] for spec in specials])
            #         maxspec = np.max([spec["vals"][ii] for spec in specials])
            #     else:
            #         minspec = spec["vals"][ii]
            #         maxspec = spec["vals"][ii]
            #     mins[ii] = min(mins[ii], minspec)
            #     maxs[ii] = max(maxs[ii], maxspec)

    deltas = (maxs - mins) / 10.0
    use_mins = mins - deltas
    use_maxs = maxs + deltas

    # cmuse = cm.get_cmap(name="tab10")

    # fig = plt.figure(constrained_layout=True)
    fig = plt.figure()
    if upper_right is None:
        gs = GridSpec(dim, dim, figure=fig)
        axs = [None] * dim * dim
        start = 0
        end = dim
        l = dim
    else:
        gs = GridSpec(dim + 1, dim + 1, figure=fig)
        axs = [None] * (dim + 1) * (dim + 1)
        start = 1
        end = dim + 1
        l = dim + 1

    # print("mins = ", mins)
    # print("maxs = ", maxs)

    for ii in range(dim):
        # print("ii = ", ii)
        axs[ii] = fig.add_subplot(gs[ii + start, ii])
        ax = axs[ii]

        # Turn everythinng off
        if ii < dim - 1:
            ax.tick_params(axis="x", bottom=False, top=False, labelbottom=False)
        else:
            ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True)
            if labels:
                ax.set_xlabel(labels[ii])

        ax.tick_params(axis="y", left=False, right=False, labelleft=False)
        ax.set_frame_on(False)

        sampii = np.concatenate([samples[kk][:, ii] for kk in range(nchains)])
        # for kk in range(nchains):
        # print("sampii == ", sampii)
        ax.hist(
            sampii,
            # ax.hist(samples[kk][:, ii],
            bins="sturges",
            density=True,
            edgecolor="black",
            stacked=True,
            range=(use_mins[ii], use_maxs[ii]),
            alpha=0.4,
        )
        if specials is not None:
            for special in specials:
                if special["vals"][ii] is not None:
                    # ax.axvline(special[ii], color='red', lw=2)
                    if "color" in special:
                        ax.axvline(special["vals"][ii], color=special["color"], lw=2)
                    else:
                        ax.axvline(special["vals"][ii], lw=2)

        ax.set_xlim((use_mins[ii] - 1e-10, use_maxs[ii] + 1e-10))

        for jj in range(ii + 1, dim):
            # print("jj = ", jj)
            axs[jj * l + ii] = fig.add_subplot(gs[jj + start, ii])
            ax = axs[jj * l + ii]

            if jj < dim - 1:
                ax.tick_params(axis="x", bottom=False, top=False, labelbottom=False)
            else:
                ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True)
                if labels:
                    ax.set_xlabel(labels[ii])
            if ii > 0:
                ax.tick_params(axis="y", left=False, right=False, labelleft=False)
            else:
                ax.tick_params(axis="y", left=True, right=False, labelleft=True)
                if labels:
                    ax.set_ylabel(labels[jj])

            ax.set_frame_on(True)

            for kk in range(nchains):
                if hist_plot is True:
                    ax.hist2d(
                        samples[kk][:, ii],
                        samples[kk][:, jj],
                        bins=nbins,
                        norm=mcolors.PowerNorm(gamma),
                        density=True,
                    )
                else:
                    ax.plot(
                        samples[kk][:, ii], samples[kk][:, jj], "o", ms=1, alpha=gamma
                    )

                # ax.hist2d(samples[kk][:, ii], samples[kk][:, jj], bins=nbins)

            if specials is not None:
                for special in specials:
                    if "color" in special:
                        ax.plot(
                            special["vals"][ii],
                            special["vals"][jj],
                            "x",
                            color=special["color"],
                            ms=2,
                            mew=2,
                        )
                    else:
                        ax.plot(
                            special["vals"][ii], special["vals"][jj], "x", ms=2, mew=2
                        )

            ax.set_xlim((use_mins[ii], use_maxs[ii]))
            ax.set_ylim((use_mins[jj] - 1e-10, use_maxs[jj] + 1e-10))

    plt.tight_layout(pad=0.01)
    if upper_right is not None:
        size_ur = int(dim / 2)

        name = upper_right["name"]
        vals = upper_right["vals"]
        if "log_transform" in upper_right:
            log_transform = upper_right["log_transform"]
        else:
            log_transform = None
        ax = fig.add_subplot(
            gs[0 : int(dim / 2), size_ur + 1 : size_ur + int(dim / 2) + 1]
        )

        lb = np.min([np.quantile(val, 0.01) for val in vals])
        ub = np.max([np.quantile(val, 0.99) for val in vals])
        for kk in range(nchains):
            if log_transform is not None:
                pv = np.log10(vals[kk])
                ra = (np.log10(lb), np.log10(ub))
            else:
                pv = vals[kk]
                ra = (lb, ub)
            ax.hist(
                pv,
                density=True,
                range=ra,
                edgecolor="black",
                stacked=True,
                bins="auto",
                alpha=0.2,
            )
        ax.tick_params(axis="x", bottom="both", top=False, labelbottom=True)
        ax.tick_params(axis="y", left="both", right=False, labelleft=False)
        ax.set_frame_on(True)
        ax.set_xlabel(name)
    plt.subplots_adjust(left=0.15, right=0.95)
    return fig, axs, gs


def plot_chains(samples, title=None):
    """Function plots the MCMC chains
    :param samples: samples from the MCMC, (Num of samples, Num of parameters)
    :return:
    """
    nchains, nparameters = samples.shape
    fig, axs = plt.subplots(nparameters, 1, figsize=(10, 10))
    for ii in range(nparameters):
        axs[ii].plot(samples[:, ii], color="k")
        axs[ii].set_ylabel(r"$\Theta_{{{}}}$".format(ii + 1))
    if title is not None:
        fig.suptitle(title)
    return fig, axs


def autocorrelation(samples, maxlag=100, step=1):
    """Compute the correlation of a set of samples
    Written by: Alex Gorodedsky
    :param samples: samples from the MCMC, (Num of samples, Num of parameters)
    :param maxlag: maximum lag to compute the correlation
    :param step: step to compute the correlation
    :return: lags, correlation
    """

    # Get the shapes
    ndim = samples.shape[1]
    nsamples = samples.shape[0]

    # Compute the mean
    mean = np.mean(samples, axis=0)

    # Compute the denominator, which is variance
    denominator = np.zeros((ndim))
    for ii in range(nsamples):
        denominator = denominator + (samples[ii, :] - mean) ** 2

    lags = np.arange(0, maxlag, step)
    autos = np.zeros((len(lags), ndim))
    for zz, lag in enumerate(lags):
        autos[zz, :] = np.zeros((ndim))
        # compute the covariance between all samples *lag apart*
        for ii in range(nsamples - lag):
            autos[zz, :] = autos[zz, :] + (samples[ii, :] - mean) * (
                samples[ii + lag, :] - mean
            )
        autos[zz, :] = autos[zz, :] / denominator
    return lags, autos
