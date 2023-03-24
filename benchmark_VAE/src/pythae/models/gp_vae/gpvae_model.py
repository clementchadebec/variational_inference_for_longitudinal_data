import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...data.datasets import BaseDataset
from ..vae import VAE
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseEncoder
from .gpvae_config import GPVAEConfig
import numpy as np


class GPVAE(VAE):
    """Wasserstein Autoencoder model.

    Args:
        model_config (WAE_MMD_Config): The Autoencoder configuration setting the main parameters
            of the model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(
        self,
        model_config: GPVAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        self.model_name = "GPVAE"

        self.time_length = model_config.time_length

        self.kernel = model_config.kernel_choice
        self.sigma = model_config.sigma
        self.length_scale = model_config.length_scale
        self.beta = model_config.beta
        self.prior = None
        self.pz_scale_inv = None
        self.pz_scale_log_abs_determinant = None

        self.kernel_scales = model_config.kernel_scales if model_config.kernel_scales is not None else [1.0]


    def forward(self, inputs: BaseDataset, **kwargs) -> ModelOutput:
        """The input data is encoded and decoded

        Args:
            inputs (BaseDataset): An instance of pythae's datasets

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """

        x = inputs["data"]
        seq_mask = inputs['seq_mask']
        pix_mask = inputs['pix_mask']
        x = x * pix_mask * seq_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        pz = self._get_prior()

        encoder_output = self.encoder(x)
        mu, log_var = encoder_output.embedding.reshape(x.shape[0], self.time_length, -1), encoder_output.log_covariance.reshape(x.shape[0], self.time_length, -1)

        qz_x = self.posterior_dist(mean=mu, log_covar=log_var)

        z = qz_x.rsample()
        z = torch.transpose(z, 1, 2)
        #print(z.shape)
        recon_x = self.decoder(z.reshape(-1, self.latent_dim))["reconstruction"]

       
        loss, recon_loss, mmd_loss = self.loss_function(recon_x, x, qz_x, pz, pix_mask=pix_mask, seq_mask=seq_mask)

        output = ModelOutput(
            loss=loss, recon_loss=recon_loss, mmd_loss=mmd_loss, recon_x=recon_x.reshape_as(x), z=z, x=x
        )

        return output

    def loss_function(self, recon_x, x, qz_x, pz, pix_mask=None, seq_mask=None):

        if self.model_config.reconstruction_loss == "mse":
            recon_loss = (
                0.5 * (
                    F.mse_loss(
                        recon_x.reshape(x.shape[0]*self.time_length, -1),
                        x.reshape(x.shape[0]*self.time_length, -1),
                        reduction="none"
                    ) * pix_mask.reshape(x.shape[0]*self.time_length, -1)
                ).sum(dim=-1).reshape(x.shape[0], -1) * seq_mask
            ).sum(dim=-1)

        elif self.model_config.reconstruction_loss == "bce":

            recon_loss = (
                (
                    F.binary_cross_entropy(
                        recon_x.reshape(x.shape[0]*self.time_length, -1),
                        x.reshape(x.shape[0]*self.time_length, -1),
                        reduction="none"
                    ) * pix_mask.reshape(x.shape[0]*self.time_length, -1)
                ).sum(dim=-1).reshape(x.shape[0], -1) * seq_mask
            ).sum(dim=-1)

        kld = self._kl_divergence(qz_x, pz).sum(dim=-1)
        #print("KLD: ", kld.mean())
        #print("recon: ", recon_loss.mean())

        return (
            recon_loss.mean(dim=0) + self.model_config.beta * kld.mean(dim=0),
            (recon_loss).mean(dim=0),
            kld.mean(dim=0),
        )

    def posterior_dist(self, mean, log_covar):

        z_size = self.latent_dim
        batch_size = mean.shape[0]
        time_length = mean.shape[1]

        
        mapped_mean = torch.transpose(mean, 2, 1)
        mapped_covar = torch.transpose(log_covar, 2, 1)


        mapped_covar = torch.nn.functional.softplus(mapped_covar)

        mapped_reshaped = mapped_covar.reshape(batch_size, z_size, 2*time_length)

        up = torch.nn.functional.pad(mapped_reshaped[:, :, time_length:-1].diag_embed(), (1, 0, 0, 1))
        diag = mapped_reshaped[:, :, :time_length].diag_embed()

        prec = diag + up


        eye = torch.eye(prec.shape[-1]).repeat((batch_size, z_size, 1, 1)).to(mean.device)
        prec = prec + eye
    
        cov_tril = torch.linalg.solve_triangular(prec.float(), eye.float(), upper=True)

        cov =  torch.transpose(cov_tril, 3, 2) @ cov_tril

        z_dist = torch.distributions.MultivariateNormal(loc=mapped_mean, covariance_matrix=cov)
        return z_dist

    def _kl_divergence(self, a, b):
        """ Batched KL divergence `KL(a || b)` for multivariate Normals.
            See https://github.com/tensorflow/probability/blob/master/tensorflow_probability
                       /python/distributions/mvn_linear_operator.py
            It's used instead of default KL class in order to exploit precomputed components for efficiency
        """

        def squared_frobenius_norm(x):
            """Helper to make KL calculation slightly more readable."""
            return (x**2).sum(dim=-1).sum(dim=-1)

        a_scale = a.covariance_matrix.cholesky()
        b_scale = b.covariance_matrix.cholesky()

        if self.pz_scale_inv is None:
            self.pz_scale_inv = torch.linalg.inv(b_scale)
            #self.pz_scale_inv = tf.where(tf.math.is_finite(self.pz_scale_inv),
            #                                self.pz_scale_inv, tf.zeros_like(self.pz_scale_inv))

       

        if self.pz_scale_log_abs_determinant is None:
            self.pz_scale_log_abs_determinant = torch.log(torch.abs(torch.det(b_scale)))

        a_shape = a_scale.shape
        if len(b.covariance_matrix.shape) == 3:
            _b_scale_inv = torch.tile(self.pz_scale_inv[None], (a_shape[0],) + tuple([1] * (len(a_shape) - 1)))
        else:
            _b_scale_inv = torch.tile(self.pz_scale_inv, (a_shape[0],) + tuple([1] * (len(a_shape) - 1)))

        b_inv_a = _b_scale_inv.float() @ a_scale

        # ~10x times faster on CPU then on GPU
        #with tf.device('/cpu:0'):
        kl_div = (self.pz_scale_log_abs_determinant - torch.log(torch.abs(torch.det(a_scale))) +
                  0.5 * (-a.loc.shape[-1] +
                  squared_frobenius_norm(b_inv_a) + squared_frobenius_norm(
                  torch.linalg.solve(b_scale.float(), (b.loc - a.loc)[..., None].float()))))
        return kl_div

    def _get_prior(self):
        if self.prior is None:
            kernel_matrices = []
            for i in range(self.kernel_scales):
                if self.kernel == "rbf":
                    kernel_matrices.append(self._rbf_kernel(self.time_length, self.length_scale / 2**i))
                elif self.kernel == "diffusion":
                    kernel_matrices.append(self._diffusion_kernel(self.time_length, self.length_scale / 2**i))
                elif self.kernel == "matern":
                    kernel_matrices.append(self._matern_kernel(self.time_length, self.length_scale / 2**i))
                elif self.kernel == "cauchy":
                    kernel_matrices.append(self._cauchy_kernel(self.time_length, self.sigma, self.length_scale / 2**i))

            # Combine kernel matrices for each latent dimension
            tiled_matrices = []
            total = 0
            for i in range(self.kernel_scales):
                if i == self.kernel_scales-1:
                    multiplier = self.latent_dim - total
                else:
                    multiplier = int(np.ceil(self.latent_dim / self.kernel_scales))
                    total += multiplier
                tiled_matrices.append(torch.tile(kernel_matrices[i].unsqueeze(0), [multiplier, 1, 1]))
            kernel_matrix_tiled = np.concatenate(tiled_matrices)
            assert len(kernel_matrix_tiled) == self.latent_dim

            self.prior = torch.distributions.MultivariateNormal(
                loc=torch.zeros(self.latent_dim, self.time_length).to(self.device),
                covariance_matrix=torch.tensor(kernel_matrix_tiled).to(self.device)
            )
        return self.prior


    def _cauchy_kernel(self, T, sigma, length_scale):
        xs = torch.range(1, T)
        xs_in = xs.unsqueeze(0)
        xs_out = xs.unsqueeze(-1)

        distance_matrix = (xs_in - xs_out) ** 2
        distance_matrix_scaled = distance_matrix / length_scale ** 2
        kernel_matrix = sigma / (distance_matrix_scaled + 1.)

        alpha = 0.001
        eye = torch.eye(xs.shape[0])
        return kernel_matrix + alpha * eye

    def _rbf_kernel(self, T, length_scale):
        xs = torch.range(1, T)
        xs_in = xs.unsqueeze(0)
        xs_out = xs.unsqueeze(1)
        distance_matrix = (xs_in - xs_out) ** 2
        distance_matrix_scaled = distance_matrix / length_scale ** 2
        kernel_matrix = torch.exp(-distance_matrix_scaled)
        return kernel_matrix

    def _diffusion_kernel(self, T, length_scale):
        assert length_scale < 0.5, "length_scale has to be smaller than 0.5 for the "\
                                "kernel matrix to be diagonally dominant"
        sigmas = torch.ones(T, T)
        a = torch.triu(sigmas, -1)
        sigmas_tridiag = (a * a.T) * length_scale
        kernel_matrix = sigmas_tridiag + torch.eye(T)*(1. - length_scale)
        return kernel_matrix

    def _matern_kernel(self, T, length_scale):
        xs = torch.range(1, T)
        xs_in = xs.unsqueeze(0)
        xs_out = xs.unsqueeze(1)
        distance_matrix = torch.abs(xs_in - xs_out)
        distance_matrix_scaled = distance_matrix / np.sqrt(length_scale)
        kernel_matrix = torch.exp(-distance_matrix_scaled)
        return kernel_matrix


    def get_nll(self, data, n_samples=1, batch_size=100):
        """
        Function computed the estimate negative log-likelihood of the model. It uses importance
        sampling method with the approximate posterior distribution. This may take a while.

        Args:
            data (torch.Tensor): The input data from which the log-likelihood should be estimated.
                Data must be of shape [Batch x n_channels x ...]
            n_samples (int): The number of importance samples to use for estimation
            batch_size (int): The batchsize to use to avoid memory issues
        """

        if n_samples <= batch_size:
            n_full_batch = 1
        else:
            n_full_batch = n_samples // batch_size
            n_samples = batch_size

        log_p = []

        for i in range(len(data)):
            x = data[i].unsqueeze(0)

            log_p_x = []

            for _ in range(n_full_batch):

                x_rep = torch.cat(batch_size * [x])

                pz = self._get_prior()

                encoder_output = self.encoder(x_rep)
                mu, log_var = encoder_output.embedding.reshape(x_rep.shape[0], self.time_length, -1), encoder_output.log_covariance.reshape(x_rep.shape[0], self.time_length, -1)

                qz_x = self.posterior_dist(mean=mu, log_covar=log_var)

                z = qz_x.rsample()
                
                log_q_z_given_x = qz_x.log_prob(z).sum(dim=-1)
                log_p_z = pz.log_prob(z).sum(dim=-1)
                
                z = torch.transpose(z, 2, 1)

                recon_x = self.decoder(z.reshape(-1, self.latent_dim))["reconstruction"]

                if self.model_config.reconstruction_loss == "mse":

                    log_p_x_given_z = (-0.5 * F.mse_loss(
                            recon_x.reshape(x_rep.shape[0]*self.time_length, -1),
                            x_rep.reshape(x_rep.shape[0]*self.time_length, -1),
                            reduction="none",
                        ).sum(dim=-1) - torch.tensor(
                            [np.prod(self.input_dim) / 2 * np.log(np.pi * 2)]
                        ).to(
                            data.device
                        )
                    ).reshape(x_rep.shape[0], -1).sum(dim=-1) # decoding distribution is assumed unit variance  N(mu, I)

                    #
                elif self.model_config.reconstruction_loss == "bce":

                    log_p_x_given_z = -F.binary_cross_entropy(
                        recon_x.reshape(x_rep.shape[0]*self.time_length, -1),
                        x_rep.reshape(x_rep.shape[0]*self.time_length, -1),
                        reduction="none",
                    ).sum(dim=-1).reshape(x_rep.shape[0], -1).sum(dim=-1)


                log_p_x.append(
                    log_p_x_given_z + log_p_z - log_q_z_given_x
                )  # log(2*pi) simplifies

            log_p_x = torch.cat(log_p_x)

            log_p.append((torch.logsumexp(log_p_x, 0) - np.log(len(log_p_x))).item())
            if i % 1000 == 0:
                print(f"Current nll at {i}: {np.mean(log_p)}")

        return np.mean(log_p)
