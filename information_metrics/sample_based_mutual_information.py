import numpy as np
from itertools import combinations
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class approx_mutual_information():
    def __init__(
            self,
            forward_model,
            eval_mean,
            eval_cov,
            model_noise_cov_scalar,
            num_outer_samples=1000,
            num_inner_samples=1000,
            log_file=None
    ):
        self.forward_model = forward_model
        self.model_noise_cov_scalar = model_noise_cov_scalar
        self.eval_mean = eval_mean
        self.eval_cov = eval_cov
        self.global_num_outer_samples = num_outer_samples
        self.global_num_inner_samples = num_inner_samples
        self.local_num_outer_samples, self.local_num_inner_samples = self.compute_local_num_samples()
        self.log_file = log_file

        self.num_parameters = self.eval_mean.shape[0]

    def compute_local_num_samples(self):
        """Function computes the local number of samples"""
        assert(self.global_num_outer_samples % size ==
               0), "Error: Outer num samples cannot be equally distribute to processors"
        local_num_outer_samples = int(self.global_num_outer_samples / size)
        local_num_inner_samples = int(self.global_num_inner_samples)
        return local_num_outer_samples, local_num_inner_samples

    def compute_model_prediction(self, theta, head=None):
        """Function computes the model prediction"""
        num_samples = theta.shape[1]
        prediction = []
        for isample in range(num_samples):
            if head is not None and isample % 50 == 0:
                self.write_log_file(head.format(isample, num_samples))
            prediction.append(self.forward_model(theta[:, isample]))
        return np.array(prediction).T

    def sample_gaussian(self, mean, cov, num_samples):
        """Function generates samples from a multivariate gaussian distribution"""
        d = mean.shape[0]
        cholesky = np.linalg.cholesky(cov)
        noise = cholesky@np.random.randn(d*num_samples).reshape(d, num_samples)
        return mean + noise

    def evaluate_gaussian_probability(self, mean, cov_scalar, samples):
        """Fuction evaluates the probability of sample from a normal distribution"""
        d, num_samples = samples.shape
        pre_exp = 1/((2*np.pi*cov_scalar)**(d/2))
        error = mean - samples
        error_norm_sq = np.linalg.norm(error, axis=0)**2
        return pre_exp*np.exp(-0.5*error_norm_sq/cov_scalar)

    def sample_likelihood(self, prediction):
        """Function samples the likelihood"""
        num_data_samples, spatial_res, num_samples = prediction.shape
        product = num_data_samples*spatial_res*num_samples
        noise = np.sqrt(self.model_noise_cov_scalar)*np.random.randn(
            product).reshape(num_data_samples, spatial_res, num_samples)
        return prediction + noise

    def evaluate_likelihood_probability(self, data, predictions):
        """Function evaluates the likelihood probability"""
        num_data_samples, spatial_res, num_samples = predictions.shape
        pre_exp = (1/((1/np.pi*self.model_noise_cov_scalar)**(spatial_res/2))
                   )**(num_data_samples)
        error = data - predictions
        error_norm_sq = np.linalg.norm(error, axis=1)
        likelihood = pre_exp * \
            np.exp(-0.5*np.sum(error_norm_sq/self.model_noise_cov_scalar, axis=0))
        return likelihood

    def select_parameter(self, itheta=None, theta_pair=None):
        """Function selectes the parameter from the entire set"""
        if theta_pair is None:
            assert (itheta is not None), "Provide the selected parameter index"
            select_parameter_mean = self.eval_mean[itheta, :].reshape(-1, 1)
            select_parameter_cov = self.eval_cov[itheta, itheta].reshape(1, 1)
        else:
            assert (
                theta_pair is not None), "Provide the selected parameter-pair index"
            select_parameter_mean = self.eval_mean[theta_pair,
                                                    :].reshape(-1, 1)
            self.display_message(
                "WARNING: Assumes the parameter are uncorreleted (a-priori)")
            select_parameter_cov = np.diag(np.diag(self.eval_cov)[theta_pair])

        return select_parameter_mean, select_parameter_cov

    def estimate_individual_mutual_information(self):
        """Function estimates the mutual information"""
        # Definitions
        individual_mutual_information = np.zeros(self.num_parameters)

        for itheta in range(self.num_parameters):
            self.write_log_file("---------------")
            self.write_log_file("Theta# : {}".format(itheta))
            self.write_log_file("---------------")

            # Compute the parameter mean and cov (selected)
            select_parameter_mean, select_parameter_cov = self.select_parameter(
                itheta)
            # Generate prior samples (Outer Loop)
            select_prior_samples = self.sample_gaussian(
                mean=select_parameter_mean,
                cov=select_parameter_cov,
                num_samples=self.local_num_outer_samples
            )

            self.write_log_file(">>> Begin Outer Evaluation for theta# : {}".format(itheta))
            outer_prior_samples = np.tile(
                self.eval_mean, (1, self.local_num_outer_samples)).astype("float")
            outer_prior_samples[itheta, :] = select_prior_samples.ravel()

            # Compute the model prediction
            outer_model_prediction = self.compute_model_prediction(
                theta=outer_prior_samples, head="Outer evaluation # {}/{}")

            # Generate samples from the likelihood (conditional distribution)
            outer_data_samples = self.sample_likelihood(
                prediction=outer_model_prediction)
            self.write_log_file(">>> End Outer Evaluation for theta# : {}".format(itheta))

            # Compute likelihood probability
            self.write_log_file(">>> Begin Likelihood computation for theta# : {}".format(itheta))
            likelihood = self.evaluate_likelihood_probability(
                data=outer_data_samples,
                predictions=outer_model_prediction
            )
            log_likelihood = np.log(likelihood)
            self.write_log_file(">>> End Likelihood computation for theta# : {}".format(itheta))

            # Compute evidence
            self.write_log_file(">>> Begin Inner Evaluation for theta# : {}".format(itheta))
            evidence = self.compute_evidence(
                outer_data_samples=outer_data_samples,
                itheta=itheta
            )

            log_evidence = np.log(evidence)
            self.write_log_file(">>> End Inner Evaluation for theta# : {}".format(itheta))

            # Compute unnormalized mutual information
            local_unnormalized_mutual_information = np.sum(
                log_likelihood - log_evidence)

            # Gather unnormalized mutual information
            comm.Barrier()
            global_unnormalized_mutual_information = comm.gather(
                local_unnormalized_mutual_information, root=0)

            if rank == 0:
                individual_mutual_information[itheta] = (
                    1/self.global_num_outer_samples)*sum(global_unnormalized_mutual_information)

        mutual_information = comm.scatter(
            [individual_mutual_information]*size, root=0)

        return mutual_information

    def compute_evidence(self, outer_data_samples, itheta):
        """Function computes the evidence"""
        # Definitions
        evidence = np.zeros(self.local_num_outer_samples)

        # Compute the parameter mean and cov (selected)
        select_parameter_mean, select_parameter_cov = self.select_parameter(
            itheta)

        # Generate prior samples (Inner Loop)
        select_prior_samples = self.sample_gaussian(
            mean=select_parameter_mean,
            cov=select_parameter_cov,
            num_samples=self.local_num_inner_samples
        )

        inner_prior_samples = np.tile(
            self.eval_mean, (1, self.local_num_inner_samples)).astype("float")
        inner_prior_samples[itheta, :] = select_prior_samples

        # Moder predictions
        inner_model_prediction = self.compute_model_prediction(
                theta=inner_prior_samples, head="Inner evaluation # {}/{}")

        for isample in range(self.local_num_outer_samples):
            data = np.expand_dims(outer_data_samples[:, :, isample], axis=-1)
            evidence_sample = self.evaluate_likelihood_probability(
                data=data,
                predictions=inner_model_prediction
            )
            evidence[isample] = (
                1/self.local_num_inner_samples)*np.sum(evidence_sample)

        return evidence

    def display_message(self, message, print_rank="root"):
        """Print function"""
        if print_rank == "root":
            if rank == 0:
                print(message)
        elif print_rank == "all":
            print(message+" at rank : {}".format(rank))
        elif rank == print_rank:
            print(message+" at rank : {}".format(rank))

    def write_log_file(self, message):
        """Function writes the log file"""
        if self.log_file is not None:
            if rank == 0:
                self.log_file.write(message+"\n")
                self.log_file.flush()
            else:
                pass
        else:
            if rank == 0:
                print(message)
            else:
                pass


class approx_conditional_mutual_information(approx_mutual_information):
    def estimate_pair_mutual_infomation(self):
        """Function estimates the mutual information between pair of parameters"""

        comb = np.array(list(combinations(np.arange(self.num_parameters), 2)))
        pair_mutual_information = np.nan * \
            np.ones((self.num_parameters, self.num_parameters))

        for ii, theta_pair in enumerate(comb):
            self.write_log_file("---------------")
            self.write_log_file("Theta pair# : {}/{}".format(ii, comb.shape[0]))
            self.write_log_file("---------------")

            # Compute the parameter mean and cov (selected)
            select_parameter_mean, select_parameter_cov = self.select_parameter(
                theta_pair=theta_pair)
            # Generate prior samples (Outer Loop)
            select_prior_samples = self.sample_gaussian(
                mean=select_parameter_mean,
                cov=select_parameter_cov,
                num_samples=self.local_num_outer_samples
            )

            self.write_log_file(">>> Begin Outer Evaluation for theta pair# : {}/{}".format(ii, comb.shape[0]))
            outer_prior_samples = np.tile(
                self.eval_mean, (1, self.local_num_outer_samples)).astype("float")
            outer_prior_samples[theta_pair, :] = select_prior_samples

            # Compute the model prediction
            outer_model_prediction = self.compute_model_prediction(
                theta=outer_prior_samples, head="Outer evaluation # {}/{}")

            # Generate samples from the likelihood (conditional distribution)
            outer_data_samples = self.sample_likelihood(
                prediction=outer_model_prediction)

            self.write_log_file(">>> End Outer Evaluation for theta pair# : {}/{}".format(ii, comb.shape[0]))

            # Compute likelihood probability
            self.write_log_file(">>> Begin Likelihood computation for theta pair# : {}/{}".format(ii, comb.shape[0]))
            likelihood = self.evaluate_likelihood_probability(
                data=outer_data_samples,
                predictions=outer_model_prediction
            )
            log_likelihood = np.log(likelihood)
            self.write_log_file(">>> End Likelihood computation for theta pair# : {}/{}".format(ii, comb.shape[0]))

            # Compute individual likelihood
            self.write_log_file(">>> Begin Individual Likelihood computation for theta pair# : {}/{}".format(ii, comb.shape[0]))
            individual_likelihood = self.compute_individual_likelihood(
                theta_pair=theta_pair,
                outer_data_samples=outer_data_samples,
                outer_prior_samples=outer_prior_samples
            )
            log_individual_likelihood = np.log(individual_likelihood)
            self.write_log_file(">>> End Individual Likelihood computation for theta pair# : {}/{}".format(ii, comb.shape[0]))

            # Compute the conditional evidence
            self.write_log_file(">>> Begin conditional evidence computation for theta pair# : {}/{}".format(ii, comb.shape[0]))
            conditional_evidence = self.compute_conditional_evidence(
                theta_pair=theta_pair,
                outer_data_samples=outer_data_samples
            )
            log_conditional_evidence = np.log(conditional_evidence)
            self.write_log_file(">>> End conditional evidence computation for theta pair# : {}/{}".format(ii, comb.shape[0]))

            # Compute unnormalized conditional mutual information
            local_unnormalized_conditional_mutual_information = np.sum(
                log_likelihood + log_conditional_evidence - log_individual_likelihood[:, 0] - log_individual_likelihood[:, 1])

            # Gather unnormalized mutual information
            comm.Barrier()
            global_unnormalized_conditional_mutual_information = comm.gather(
                local_unnormalized_conditional_mutual_information, root=0)

            if rank == 0:
                pair_mutual_information[theta_pair[0], theta_pair[1]] = (
                    1/self.global_num_outer_samples)*sum(global_unnormalized_conditional_mutual_information)
                pair_mutual_information[theta_pair[1], theta_pair[0]] = pair_mutual_information[theta_pair[0], theta_pair[1]]

        conditional_mutual_information = comm.scatter(
            [pair_mutual_information]*size, root=0)

        return conditional_mutual_information

    def compute_individual_likelihood(self, theta_pair, outer_data_samples, outer_prior_samples):
        """Function computes the individual likelihood"""
        # Definitions
        individual_likelihood = np.zeros((self.local_num_outer_samples, 2))

        for ii, itheta in enumerate(theta_pair):
            self.write_log_file("Individual likelihood pair# : {}".format(ii))
            sample_theta = theta_pair[np.arange(2) == ii]
            fixed_theta = theta_pair[np.arange(2) != ii]

            parameter_mean_selected = self.eval_mean[itheta, :].reshape(-1, 1)
            parameter_cov_selected = self.eval_cov[itheta, itheta].reshape(
                1, 1)

            select_parameter_mean, select_parameter_cov = self.select_parameter(
                itheta=itheta)

            # Generate the prior samples (inner loop)
            select_prior_samples = self.sample_gaussian(
                mean=select_parameter_mean,
                cov=select_parameter_cov,
                num_samples=self.local_num_inner_samples
            )

            for isample in range(self.local_num_outer_samples):
                self.write_log_file("$$ Individual Likelihood outer sample #{} $$".format(isample))
                data = np.expand_dims(
                    outer_data_samples[:, :, isample], axis=-1)
                outer_fixed_sample = outer_prior_samples[fixed_theta, isample]

                prior_samples = np.tile(
                    self.eval_mean, (1, self.local_num_inner_samples)).astype("float")
                prior_samples[sample_theta, :] = select_prior_samples
                prior_samples[fixed_theta, :] = outer_fixed_sample

                model_prediction = self.compute_model_prediction(
                    theta=prior_samples, head="Individual likelihood evaluation #{}/{}")

                individual_likelihood_samples = self.evaluate_likelihood_probability(
                    data=data, predictions=model_prediction)
                individual_likelihood[isample, ii] = (
                    1/self.local_num_inner_samples)*np.sum(individual_likelihood_samples)

        return individual_likelihood

    def compute_conditional_evidence(self, theta_pair, outer_data_samples):
        """Function comptues the conditional evidence"""
        # Definitions
        evidence = np.zeros(self.local_num_outer_samples)

        # Compute the parameter mean and cov (selected)
        select_parameter_mean, select_parameter_cov = self.select_parameter(
            theta_pair=theta_pair)

        select_prior_samples = self.sample_gaussian(
            mean=select_parameter_mean,
            cov=select_parameter_cov,
            num_samples=self.local_num_inner_samples
        )

        inner_prior_samples = np.tile(
            self.eval_mean, (1, self.local_num_inner_samples)).astype("float")
        inner_prior_samples[theta_pair, :] = select_prior_samples

        # Moder predictions
        inner_model_prediction = self.compute_model_prediction(
            theta=inner_prior_samples, head="Conditional evidence evaluation #{}/{}")

        for isample in range(self.local_num_outer_samples):
            data = np.expand_dims(outer_data_samples[:, :, isample], axis=-1)
            evidence_sample = self.evaluate_likelihood_probability(
                data=data,
                predictions=inner_model_prediction
            )
            evidence[isample] = (
                1/self.local_num_inner_samples)*np.sum(evidence_sample)

        return evidence
