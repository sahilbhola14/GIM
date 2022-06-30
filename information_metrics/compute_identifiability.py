# Developer : Sahil Bhola
# Date : 6/24/2022
# Description : Model computes the identifiabiliy (practical) of the model parameters
# using information theory (see readme for more information (pun intended))

import numpy as np
import matplotlib.pyplot as plt
import sys
from itertools import combinations
sys.path.append("../examples/linear_gaussian")
from quadrature import *


class mutual_information():
    def __init__(
            self,
            forward_model,
            prior_mean,
            prior_cov,
            model_noise_cov_scalar,
            num_outer_samples,
            num_inner_samples,
            ytrain,
            log_file=None
    ):
        self.forward_model = forward_model
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.model_noise_cov_scalar = model_noise_cov_scalar
        self.num_outer_samples = num_outer_samples
        self.num_inner_samples = num_inner_samples
        self.ytrain = ytrain
        self.log_file = log_file
        self.num_parameters = self.prior_mean.shape[0]

        self.outer_prior_samples = np.zeros(
            (self.num_parameters, self.num_outer_samples))
        self.outer_data_samples = np.zeros(
            (self.num_outer_samples, )+self.ytrain.shape).T
        self.outer_model_prediction = np.zeros(
            (self.num_outer_samples, )+self.ytrain.shape).T

    def sample_gaussian(self, mean, cov, num_samples):
       """Function samples the gaussian"""
        assert(mean.shape[1] >= 1), "incompatible mean shape"
        # Definitions
        d = mean.shape[0]
        L = np.linalg.cholesky(cov+np.eye(d)*1e-8)
        noise = L@np.random.randn(d*num_samples).reshape(d, num_samples)
        return mean + noise

    def evaluate_gaussian_probability(self, mean, cov, samples):
        """Function evaluates the proabability of the samples for a gaussian distribution"""
        # Definitions
        d, num_samples = samples.shape
        try_batches = [200, 100, 50, 10]

        if num_samples > 1:
            working_batches = []
            for itry in try_batches:
                if num_samples % itry == 0:
                    working_batches.append(itry)
            if len(working_batches) == 0:
                num_batches = 1
            else:
                num_batches = max(working_batches)
        else:
            num_batches = 1

        batch_size = int(num_samples/num_batches)

        pre_exp = 1/(((2*np.pi)**(d/2))*np.linalg.det(cov)**(0.5))
        error_norm_sq = np.zeros(num_samples)
        for ibatch in range(num_batches):
            samples_mini_batch = samples[:,
                                         (ibatch)*batch_size: (ibatch+1)*batch_size]
            error = samples_mini_batch - mean

            error_norm_sq[(ibatch)*batch_size:(ibatch+1) *
                          batch_size] = np.diag(error.T@cov@error)

        # if num_samples > 1000:
        #     error_norm_sq = np.zeros(num_samples)
        #     # Create mini-batches
        #     assert(batch_size % num_samples == 0), "Modify the batch size"
        #     num_batches = int(num_samples/batch_size)
        #     for ibatch in range(num_batches):
        #         samples_mini_batch = samples[:,
        #                                      (ibatch)*batch_size: (ibatch+1)*batch_size]
        #         error = samples_mini_batch - mean

        #         error_norm_sq[(ibatch)*batch_size:(ibatch+1) *
        #                       batch_size] = np.diag(error.T@cov@error)

        # else:
        #     error = samples - mean
        #     error_norm_sq = np.diag(error.T@cov@error)

        exp_term = np.exp(-0.5*error_norm_sq)
        return pre_exp*exp_term

    def sample_prior(self, num_samples):
      """Function samples the prior distribution (currently only gaussian)"""
        prior_samples = self.sample_gaussian(
            mean=self.prior_mean,
            cov=self.prior_cov,
            num_samples=num_samples
        )

        return prior_samples

    def compute_model_prediction(self, theta, write_label_format=None):
        """Function computes the model prediction"""
        num_samples = theta.shape[1]
        prediction = []
        for isample in range(num_samples):
            # if head is not None and isample % 50 == 0:
            # self.write_log_file(head.format(isample, num_samples))
            prediction.append(self.forward_model(theta[:, isample]))
        return np.array(prediction).T

    def sample_likelihood(self, prior_samples):
        """Function samples the likelihood essentially resulting in samples
        from the joint distribution p(y, theta)"""
        model_prediction = self.compute_model_prediction(theta=prior_samples)
        product = np.prod(model_prediction.shape)
        noise = np.sqrt(self.model_noise_cov_scalar) * \
            np.random.randn(product).reshape(model_prediction.shape)

        return model_prediction + noise, model_prediction

    def compute_mutual_information_via_mc(self, use_quadrature=False):
        """Function computes the mutual information using mc"""
        # Prior samples
        self.outer_prior_samples = self.sample_prior(
            num_samples=self.num_outer_samples)
        # Model prediction
        self.outer_data_samples, self.outer_model_prediction = self.sample_likelihood(
            prior_samples=self.outer_prior_samples)
        # Likelihood probability
        likelihood = self.evaluate_likelihood_proabability(
            data=self.outer_data_samples,
            model_prediction=self.outer_model_prediction
        )
        log_likelihood = np.log(likelihood)
        evidence = self.estimate_evidence_probability(
            data=self.outer_data_samples,
            use_quadrature=use_quadrature,
            quadrature_rule="gaussian"
            # quadrature_rule="unscented"
        )
        log_evidence = np.log(evidence)

        estimated_mutual_information = (
            1/self.num_outer_samples)*np.sum(log_likelihood - log_evidence)
        print("Estimated mutual information : {}".format(
            estimated_mutual_information))

    def evaluate_likelihood_proabability(self, data, model_prediction):
        """Function evaluates the likelihood probability"""
        num_data_samples, spatial_res, num_samples = data.shape

        error = data - model_prediction
        error_norm_sq = np.linalg.norm(error, axis=1)**2
        pre_exp = 1 / ((2*np.pi*self.model_noise_cov_scalar)**(spatial_res/2))

        likelihood = pre_exp * \
            np.exp(-0.5*np.sum(error_norm_sq / self.model_noise_cov_scalar, axis=0))
        return likelihood

    def estimate_evidence_probability(self, data, use_quadrature=False, quadrature_rule="unscented"):
        """Function estimates the evidence probability"""
        num_data_samples, spatial_res, num_samples = data.shape
        evidence_prob = np.zeros(num_samples)

        pre_exp = 1 / ((2*np.pi*self.model_noise_cov_scalar)**(spatial_res/2))
        if use_quadrature is False:
            inner_prior_samples = self.sample_prior(
                num_samples=self.num_inner_samples)
            inner_model_prediction = self.compute_model_prediction(
                theta=inner_prior_samples)

            for isample in range(self.num_outer_samples):
                outer_sample = np.expand_dims(data[:, :, isample], axis=-1)
                error = outer_sample - inner_model_prediction
                error_norm_sq = np.linalg.norm(error, axis=1)**2
                sample_evidence_estimates = pre_exp * \
                    np.exp(-0.5*np.sum(error_norm_sq /
                                       self.model_noise_cov_scalar, axis=0))
                evidence_prob[isample] = (
                    1/self.num_inner_samples)*np.sum(sample_evidence_estimates)

        if use_quadrature is True:
            for isample in range(self.num_outer_samples):
                outer_sample = np.expand_dims(data[:, :, isample], axis=-1)
                def integrand(x): return self.quadrature_integrand_function(
                    x, outer_sample=outer_sample)

                if quadrature_rule == "unscented":
                    unscented_quad = unscented_quadrature(
                        mean=self.prior_mean,
                        cov=self.prior_cov,
                        integrand=integrand
                    )
                    evidence_mean, evidence_cov = unscented_quad.compute_integeral()
                    evidence_prob[isample] = evidence_mean + \
                        np.sqrt(evidence_cov)*np.random.randn(1)
                    # evidence_prob[isample] = evidence_mean
                elif quadrature_rule == "gaussian":
                    gh = gauss_hermite_quadrature(
                        mean=self.prior_mean,
                        cov=self.prior_cov,
                        integrand=integrand,
                        num_points=60
                    )
                    evidence_prob[isample] = gh.compute_integeral()
                else:
                    raise ValueError("Invalid quadrature rule")

        return evidence_prob

    def quadrature_integrand_function(self, parameter_samples, outer_sample):
        """Function returns the likelihood probability"""
        num_data_samples, spatial_res, num_samples = outer_sample.shape
        model_prediction = self.compute_model_prediction(parameter_samples)
        pre_exp = 1 / ((2*np.pi*self.model_noise_cov_scalar)**(spatial_res/2))

        error = outer_sample - model_prediction
        error_norm_sq = np.linalg.norm(error, axis=1)**2
        sample_evidence_estimates = pre_exp * \
            np.exp(-0.5*np.sum(error_norm_sq /
                               self.model_noise_cov_scalar, axis=0))
        return sample_evidence_estimates.reshape(1, -1)


class conditional_mutual_information(mutual_information):
    def compute_individual_parameter_data_mutual_information_via_mc(self, compute_outer_samples=False, use_quadrature=False):
        """Function computes the conditional mutual information for each parameter via mc."""
        # Definitions
        individual_mutual_information = np.zeros(self.num_parameters)

        if compute_outer_samples is True:
            # Prior samples
            self.outer_prior_samples = self.sample_prior(
                num_samples=self.num_outer_samples)

            # Model prediction
            self.outer_data_samples, self.outer_model_prediction = self.sample_likelihood(
                prior_samples=self.outer_prior_samples)

        # Likelihood probability
        likelihood = self.evaluate_likelihood_proabability(
            data=self.outer_data_samples,
            model_prediction=self.outer_model_prediction
        )
        log_likelihood = np.log(likelihood)

        for iparameter in range(self.num_parameters):
            conditional_evidence = self.estimate_conditional_evidence(
                data=self.outer_data_samples,
                parameter_pair=np.array([iparameter]),
                use_quadrature=use_quadrature,
                quadrature_rule="gaussian"
            )
            log_conditional_evidence = np.log(conditional_evidence)
            individual_mutual_information[iparameter] = (
                1/self.num_outer_samples)*np.sum(log_likelihood - log_conditional_evidence)

        print("Estimated individual mutual information I(theta_i;Y|theta_not_i) : {}".format(
            individual_mutual_information))

    def compute_pair_parameter_data_mutual_information_via_mc(self, compute_outer_samples=False, use_quadrature=False):
       """Function computes the pair wise mutual information between the parameter and the data"""
        if compute_outer_samples is True:
            # Prior samples
            self.outer_prior_samples = self.sample_prior(
                num_samples=self.num_outer_samples)

            # Model prediction
            self.outer_data_samples, self.outer_model_prediction = self.sample_likelihood(
                prior_samples=self.outer_prior_samples)

        # Likelihood probability
        likelihood = self.evaluate_likelihood_proabability(
            data=self.outer_data_samples,
            model_prediction=self.outer_model_prediction
        )
        log_likelihood = np.log(likelihood)

        parameter_combinations = np.array(
            list(combinations(np.arange(self.num_parameters), 2)))

        for iparameter in parameter_combinations:
            print(iparameter)
            single_sample_likelihood = self.compute_single_sample_likelihood_prob(
                    data=self.outer_data_samples,
                    parameter_pair=iparameter,
                    use_quadrature=use_quadrature,
                    quadrature_rule="gaussian"
                    )
            log_single_sample_likelihood = np.log(single_sample_likelihood)

            pair_sample_likelihood = self.compute_pair_sample_likelihood_prob(
                data=self.outer_data_samples,
                parameter_pair=iparameter,
                use_quadrature=use_quadrature,
                quadrature_rule="gaussian"
            )
            log_pair_sample_likelihood = np.log(pair_sample_likelihood)
            conditional_mutual_information = (1/self.num_outer_samples)*np.sum(log_likelihood + log_pair_sample_likelihood - np.sum(log_single_sample_likelihood, axis=1))
            print("I(theta_{};theta_{} | y, theta_[not selected]) : {}".format(iparameter[0], iparameter[1], conditional_mutual_information))

    def compute_single_sample_likelihood_prob(self, data, parameter_pair, use_quadrature=False, quadrature_rule="unscented"):
        """Function comptues the single sample likelihood, p(y|theta_i/j, theta_k)"""
        num_data_samples, spatial_res, num_samples = data.shape
        pre_exp = 1 / ((2*np.pi*self.model_noise_cov_scalar)**(spatial_res/2))
        likelihood_prob = np.zeros((self.num_outer_samples, 2))
        for ii in range(2):
            sample_parameter_id = np.array([parameter_pair[ii]])
            fixed_parameter_id = np.arange(
                self.num_parameters) != sample_parameter_id
            sample_parameter_mean, sample_parameter_cov = self.select_parameter(
                sample_parameter_id)

            if use_quadrature is True:
                for isample in range(self.num_outer_samples):
                    outer_sample = np.expand_dims(data[:, :, isample], axis=-1)
                    fixed_parameters = self.outer_prior_samples[fixed_parameter_id, isample].reshape(
                                            -1, 1)

                    def integrand(x):
                        return self.quadrature_integrand_function_individual_sample_likelihood(
                                parameter_samples=x,
                                outer_sample=outer_sample,
                                fixed_parameters=fixed_parameters,
                                sample_parameter_id=sample_parameter_id,
                                fixed_parameter_id=fixed_parameter_id
                                )

                    if quadrature_rule == "unscented":
                        unscented_quad = unscented_quadrature(
                            mean=sample_parameter_mean,
                            cov=sample_parameter_cov,
                            integrand=integrand
                        )
                        evidence_mean, evidence_cov = unscented_quad.compute_integeral()
                        likelihood_prob[isample, ii] = evidence_mean + \
                            np.sqrt(evidence_cov)*np.random.randn(1)

                    elif quadrature_rule == "gaussian":
                        gh = gauss_hermite_quadrature(
                            mean=sample_parameter_mean,
                            cov=sample_parameter_cov,
                            integrand=integrand,
                            num_points=60
                        )
                        likelihood_prob[isample, ii] = gh.compute_integeral()
                    else:
                        raise ValueError("Invalid quadrature rule")

            else:
                inner_parameter_samples = self.sample_gaussian(
                    mean=sample_parameter_mean,
                    cov=sample_parameter_cov,
                    num_samples=self.num_inner_samples
                )

                for isample in range(self.num_outer_samples):
                    outer_sample = np.expand_dims(data[:, :, isample], axis=-1)
                    fixed_parameters = self.outer_prior_samples[fixed_parameter_id, isample].reshape(
                        -1, 1)
                    eval_parameters = np.zeros(
                        (self.num_parameters, self.num_inner_samples))
                    eval_parameters[sample_parameter_id,
                                    :] = inner_parameter_samples
                    eval_parameters[fixed_parameter_id, :] = np.tile(
                        fixed_parameters, (1, self.num_inner_samples))

                    inner_model_prediction = self.compute_model_prediction(
                        theta=eval_parameters)

                    error = outer_sample - inner_model_prediction
                    error_norm_sq = np.linalg.norm(error, axis=1)**2
                    sample_evidence_estimates = pre_exp * \
                        np.exp(-0.5*np.sum(error_norm_sq /
                                           self.model_noise_cov_scalar, axis=0))
                    likelihood_prob[isample, ii] = (
                        1/self.num_inner_samples)*np.sum(sample_evidence_estimates)

        return likelihood_prob

    def quadrature_integrand_function_individual_sample_likelihood(self, parameter_samples, outer_sample, fixed_parameters, sample_parameter_id, fixed_parameter_id):
        num_eval_samples = parameter_samples.shape[1]
        num_data_samples, spatial_res, num_samples = outer_sample.shape
        eval_samples = np.zeros((self.num_parameters, num_eval_samples))

        eval_samples[sample_parameter_id, :] = parameter_samples
        eval_samples[fixed_parameter_id, :] = np.tile(
            fixed_parameters, (1, num_eval_samples))
        model_prediction = self.compute_model_prediction(theta=eval_samples)
        pre_exp = 1 / ((2*np.pi*self.model_noise_cov_scalar)**(spatial_res/2))
        error = outer_sample - model_prediction
        error_norm_sq = np.linalg.norm(error, axis=1)**2
        sample_evidence_estimates = pre_exp * \
            np.exp(-0.5*np.sum(error_norm_sq /
                               self.model_noise_cov_scalar, axis=0))
        return sample_evidence_estimates.reshape(1, -1)

    def compute_pair_sample_likelihood_prob(self, data, parameter_pair, use_quadrature=False, quadrature_rule="unscented"):
        """Function computes the pair sample likelihood, p(y|theta_k)"""
        num_data_samples, spatial_res, num_samples = data.shape
        pre_exp = 1 / ((2*np.pi*self.model_noise_cov_scalar)**(spatial_res/2))
        likelihood_prob = np.zeros(self.num_outer_samples)
        sample_parameter_mean, sample_parameter_cov = self.select_parameter(
            parameter_pair=parameter_pair)

        def get_fixed_parameter_id(x):
            parameter_id = np.arange(self.num_parameters)
            return ~np.logical_or(parameter_id == x[0], parameter_id == x[1])

        fixed_parameter_id = get_fixed_parameter_id(parameter_pair)
        if use_quadrature is True:
            for isample in range(self.num_outer_samples):
                outer_sample = np.expand_dims(data[:, :, isample], axis=-1)
                fixed_parameters = self.outer_prior_samples[fixed_parameter_id, isample].reshape(
                                        -1, 1)

                def integrand(x):
                    return self.quadrature_integrand_function_pair_sample_likelihood(
                            parameter_samples=x,
                            outer_sample=outer_sample,
                            fixed_parameters=fixed_parameters,
                            parameter_pair=parameter_pair,
                            )

                if quadrature_rule == "unscented":
                    unscented_quad = unscented_quadrature(
                        mean=sample_parameter_mean,
                        cov=sample_parameter_cov,
                        integrand=integrand
                    )
                    evidence_mean, evidence_cov = unscented_quad.compute_integeral()
                    likelihood_prob[isample] = evidence_mean + \
                        np.sqrt(evidence_cov)*np.random.randn(1)

                elif quadrature_rule == "gaussian":
                    gh = gauss_hermite_quadrature(
                        mean=sample_parameter_mean,
                        cov=sample_parameter_cov,
                        integrand=integrand,
                        num_points=60
                    )
                    likelihood_prob[isample] = gh.compute_integeral()
                else:
                    raise ValueError("Invalid quadrature rule")
        else:
            inner_parameter_samples = self.sample_gaussian(
                mean=sample_parameter_mean,
                cov=sample_parameter_cov,
                num_samples=self.num_inner_samples
            )
            for isample in range(self.num_outer_samples):
                outer_sample = np.expand_dims(data[:, :, isample], axis=-1)
                fixed_parameters = self.outer_prior_samples[fixed_parameter_id, isample].reshape(
                    -1, 1)
                eval_parameters = np.zeros(
                    (self.num_parameters, self.num_inner_samples))
                eval_parameters[parameter_pair, :] = inner_parameter_samples
                eval_parameters[fixed_parameter_id, :] = np.tile(
                    fixed_parameters, (1, self.num_inner_samples))

                inner_model_prediction = self.compute_model_prediction(
                    theta=eval_parameters)

                error = outer_sample - inner_model_prediction
                error_norm_sq = np.linalg.norm(error, axis=1)**2
                sample_evidence_estimates = pre_exp * \
                    np.exp(-0.5*np.sum(error_norm_sq /
                                       self.model_noise_cov_scalar, axis=0))
                likelihood_prob[isample] = (
                    1/self.num_inner_samples)*np.sum(sample_evidence_estimates)

        return likelihood_prob

    def quadrature_integrand_function_pair_sample_likelihood(self, parameter_samples, outer_sample, parameter_pair, fixed_parameters):
        num_eval_samples = parameter_samples.shape[1]
        num_data_samples, spatial_res, num_samples = outer_sample.shape
        eval_samples = np.zeros((self.num_parameters, num_eval_samples))
        def get_fixed_parameter_id(x):
            parameter_id = np.arange(self.num_parameters)
            return ~np.logical_or(parameter_id == x[0], parameter_id == x[1])

        fixed_parameter_id = get_fixed_parameter_id(parameter_pair)
        eval_samples[parameter_pair, :] = parameter_samples
        eval_samples[fixed_parameter_id, :] = np.tile(
            fixed_parameters, (1, num_eval_samples))
        model_prediction = self.compute_model_prediction(theta=eval_samples)
        pre_exp = 1 / ((2*np.pi*self.model_noise_cov_scalar)**(spatial_res/2))
        error = outer_sample - model_prediction
        error_norm_sq = np.linalg.norm(error, axis=1)**2
        sample_evidence_estimates = pre_exp * \
            np.exp(-0.5*np.sum(error_norm_sq /
                               self.model_noise_cov_scalar, axis=0))
        return sample_evidence_estimates.reshape(1, -1)

    def estimate_conditional_evidence(self, data, parameter_pair, use_quadrature=False, quadrature_rule="unscented"):
        """Function estimates the conditional evidence"""
        num_data_samples, spatial_res, num_samples = data.shape
        evidence_prob = np.zeros(num_samples)
        pre_exp = 1 / ((2*np.pi*self.model_noise_cov_scalar)**(spatial_res/2))
        sample_parameter_mean, sample_parameter_cov = self.select_parameter(
            parameter_pair)

        def get_fixed_parameter_id(x): return np.arange(
            self.num_parameters) != x

        if use_quadrature is True:
            for isample in range(self.num_outer_samples):
                outer_sample = np.expand_dims(data[:, :, isample], axis=-1)
                fixed_parameters = self.outer_prior_samples[get_fixed_parameter_id(
                    parameter_pair), isample].reshape(-1, 1)

                def integrand(x): return self.quadrature_integrand_function_conditional_evidence(
                    parameter_samples=x, outer_sample=outer_sample, fixed_parameters=fixed_parameters, parameter_pair=parameter_pair)

                if quadrature_rule == "unscented":
                    unscented_quad = unscented_quadrature(
                        mean=sample_parameter_mean,
                        cov=sample_parameter_cov,
                        integrand=integrand
                    )
                    evidence_mean, evidence_cov = unscented_quad.compute_integeral()
                    evidence_prob[isample] = evidence_mean + \
                        np.sqrt(evidence_cov)*np.random.randn(1)

                elif quadrature_rule == "gaussian":
                    gh = gauss_hermite_quadrature(
                        mean=sample_parameter_mean,
                        cov=sample_parameter_cov,
                        integrand=integrand,
                        num_points=60
                    )
                    evidence_prob[isample] = gh.compute_integeral()
                else:
                    raise ValueError("Invalid quadrature rule")

        else:
            inner_parameter_samples = self.sample_gaussian(
                mean=sample_parameter_mean,
                cov=sample_parameter_cov,
                num_samples=self.num_inner_samples
            )

            for isample in range(self.num_outer_samples):
                outer_sample = np.expand_dims(data[:, :, isample], axis=-1)
                fixed_parameters = self.outer_prior_samples[get_fixed_parameter_id(
                    parameter_pair), isample].reshape(-1, 1)
                eval_parameters = np.zeros(
                    (self.num_parameters, self.num_outer_samples))
                eval_parameters[parameter_pair, :] = inner_parameter_samples
                eval_parameters[get_fixed_parameter_id(parameter_pair), :] = np.tile(
                    fixed_parameters, (1, self.num_inner_samples))
                inner_model_prediction = self.compute_model_prediction(
                    theta=eval_parameters)

                error = outer_sample - inner_model_prediction
                error_norm_sq = np.linalg.norm(error, axis=1)**2
                sample_evidence_estimates = pre_exp * \
                    np.exp(-0.5*np.sum(error_norm_sq /
                                       self.model_noise_cov_scalar, axis=0))
                evidence_prob[isample] = (
                    1/self.num_inner_samples)*np.sum(sample_evidence_estimates)

        return evidence_prob

    def quadrature_integrand_function_conditional_evidence(self, parameter_samples, outer_sample, fixed_parameters, parameter_pair):
        num_eval_samples = parameter_samples.shape[1]
        num_data_samples, spatial_res, num_samples = outer_sample.shape
        eval_samples = np.zeros((self.num_parameters, num_eval_samples))
        def get_fixed_parameter_id(x): return np.arange(
            self.num_parameters) != x

        eval_samples[parameter_pair, :] = parameter_samples
        eval_samples[get_fixed_parameter_id(parameter_pair), :] = np.tile(
            fixed_parameters, (1, num_eval_samples))
        model_prediction = self.compute_model_prediction(theta=eval_samples)
        pre_exp = 1 / ((2*np.pi*self.model_noise_cov_scalar)**(spatial_res/2))

        error = outer_sample - model_prediction
        error_norm_sq = np.linalg.norm(error, axis=1)**2
        sample_evidence_estimates = pre_exp * \
            np.exp(-0.5*np.sum(error_norm_sq /
                               self.model_noise_cov_scalar, axis=0))
        return sample_evidence_estimates.reshape(1, -1)

    def select_parameter(self, parameter_pair):
        mean = self.prior_mean[parameter_pair].reshape(
            parameter_pair.shape[0], 1)
        cov = np.diag(np.diag(self.prior_cov)[parameter_pair])
        return mean, cov

    def build_complete_parameter_sample(self, prior_samples, parameter_pair):
        """Function builds the parameter sample"""
        d, N = prior_samples.shape
        parameters = np.tile(self.prior_mean, (1, N)).astype('float')
        parameters[parameter_pair, :] = prior_samples
        return parameters
