import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...data.datasets import BaseDataset
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseEncoder
from ..normalizing_flows import IAF, IAFConfig
from ..vae import VAE
from .lcvae_config import LCVAE_Config


class LCVAE(VAE):
    """Longitudinal Variational Auto Encoder with Inverse Autoregressive Flows
    (:class:`~pythae.models.normalizing_flows.IAF`).

    Args:
        model_config(VAE_IAF_Config): The Variational Autoencoder configuration seting the main
            parameters of the model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(
        self,
        model_config: LCVAE_Config,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        self.model_name = "VAE_IAF"

        self.n_obs = model_config.n_obs_per_ind
        self.warmup = model_config.warmup
        self.context_dim = model_config.context_dim

        self.flows = nn.ModuleList()

        iaf_config = IAFConfig(
            input_dim=(model_config.latent_dim,),
            n_made_blocks=model_config.n_made_blocks,
            n_hidden_in_made=model_config.n_hidden_in_made,
            hidden_size=model_config.hidden_size,
            include_batch_norm=False,
            context_dim=model_config.context_dim
        )

        for i in range(self.n_obs - 1):
            self.flows.append(IAF(iaf_config))
        

    def forward(self, inputs: BaseDataset, **kwargs):
        """
        The VAE NF model

        Args:
            inputs (BaseDataset): The training dataset with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """

        x = inputs["data"]
        epoch = kwargs.pop("epoch", 1)

        t = torch.linspace(0, 1, self.n_obs).repeat(x.shape[0], 1).to(x.device)
        x_t = torch.cat((t.unsqueeze(-1), x.reshape((x.shape[0], self.n_obs,) + x.shape[2:])), dim=-1)


        encoder_output = self.encoder(x_t.reshape((x.shape[0]*self.n_obs,-1)))#, torch.arange(0, self.n_obs).to(x.device).repeat(x.shape[0]).unsqueeze(-1) / self.n_obs)
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        std = torch.exp(0.5 * log_var)
        z, _ = self._sample_gauss(mu, std)

        z_seq = z

        
        z = torch.cat((t.unsqueeze(-1), z.reshape(x.shape[0], -1, self.latent_dim)), dim=-1)

        recon_x = self.decoder(z.reshape(-1, self.latent_dim+1)).reconstruction#_seq, torch.arange(0, self.n_obs).to(x.device).repeat(x.shape[0]).unsqueeze(-1) / self.n_obs)["reconstruction"] # [B*n_obs x input_dim]

        loss, recon_loss, kld = super().loss_function(recon_x, x.reshape((x.shape[0]*self.n_obs,) + x.shape[2:]), mu, log_var, z)


        output = ModelOutput(
            reconstruction_loss=recon_loss,
            reg_loss=kld,
            loss=loss,
            recon_x=recon_x.reshape_as(x),
            z=z,
            z_seq=z_seq
        )

        return output

    def loss_function(self, recon_x, x, mu, log_var, z_seq, z_vi_index, log_abs_det_jac):

        if self.model_config.reconstruction_loss == "mse":

            recon_loss = F.mse_loss(
                recon_x.reshape(x.shape[0]*self.n_obs, -1),
                x.reshape(x.shape[0]*self.n_obs, -1),
                reduction="none",
            ).sum(dim=-1).reshape(x.shape[0], -1).mean(dim=-1)

        elif self.model_config.reconstruction_loss == "bce":

            recon_loss = F.binary_cross_entropy(
                recon_x.reshape(x.shape[0]*self.n_obs, -1),
                x.reshape(x.shape[0]*self.n_obs, -1),
                reduction="none",
            ).sum(dim=-1).reshape(x.shape[0], -1).mean(dim=-1)

        z0 = z_seq[:, 0]

        # starting gaussian log-density
        log_prob_z_vi_index = (
            -0.5 * (log_var + torch.pow(z_vi_index - mu, 2) / torch.exp(log_var))
        ).sum(dim=1)

        # prior log-density
        log_prior_z_vi_index = (-0.5 * torch.pow(z0, 2)).sum(dim=1) + log_abs_det_jac

        KLD = log_prob_z_vi_index - log_prior_z_vi_index

        return (recon_loss + KLD).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)

    def generate(self, z):


        z_for = z
        z_seq = [z]

        rec = []
        for i in range(self.n_obs):
            t = i * torch.ones(z.shape[0]).to(z.device)

            
            z_t = torch.cat((t.unsqueeze(-1), z.reshape(z.shape[0], self.latent_dim)), dim=-1)

            rec.append(self.decoder(z_t).reconstruction.detach().cpu())
            

        rec = torch.cat(rec)

        #t = torch.linspace(0, 1, self.n_obs).repeat(z.shape[0], 1).to(z.device)
        #z = torch.cat((t.unsqueeze(-1), z_seq.reshape(z.shape[0], -1, self.latent_dim)), dim=-1)
        return rec
    #def infer_traj(self, x):

    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

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
        normal = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.model_config.latent_dim).to(data.device),
            covariance_matrix=torch.eye(self.model_config.latent_dim).to(data.device),
        )

        if n_samples <= batch_size:
            n_full_batch = 1
        else:
            n_full_batch = n_samples // batch_size
            n_samples = batch_size

        log_p = []

        for i in range(len(data)):
            x = data[i].unsqueeze(0)

            log_p_x = []

            for j in range(n_full_batch):

                x_rep = torch.cat(batch_size * [x])

                encoder_output = self.encoder(x_rep)
                mu, log_var = encoder_output.embedding, encoder_output.log_covariance

                std = torch.exp(0.5 * log_var)
                z, eps = self._sample_gauss(mu, std)

                z0 = z

                # Pass it through the Normalizing flows
                flow_output = self.iaf_flow.inverse(z)  # sampling

                z = flow_output.out
                log_abs_det_jac = flow_output.log_abs_det_jac

                log_q_z_given_x = (
                    -0.5 * (log_var + torch.pow(z0 - mu, 2) / torch.exp(log_var))
                ).sum(dim=1) - log_abs_det_jac
                log_p_z = -0.5 * (z ** 2).sum(dim=-1)

                recon_x = self.decoder(z)["reconstruction"]

                if self.model_config.reconstruction_loss == "mse":

                    log_p_x_given_z = -0.5 * F.mse_loss(
                        recon_x.reshape(x_rep.shape[0], -1),
                        x_rep.reshape(x_rep.shape[0], -1),
                        reduction="none",
                    ).sum(dim=-1) - torch.tensor(
                        [np.prod(self.input_dim) / 2 * np.log(np.pi * 2)]
                    ).to(
                        data.device
                    )  # decoding distribution is assumed unit variance  N(mu, I)

                elif self.model_config.reconstruction_loss == "bce":

                    log_p_x_given_z = -F.binary_cross_entropy(
                        recon_x.reshape(x_rep.shape[0], -1),
                        x_rep.reshape(x_rep.shape[0], -1),
                        reduction="none",
                    ).sum(dim=-1)

                log_p_x.append(
                    log_p_x_given_z + log_p_z - log_q_z_given_x
                )  # log(2*pi) simplifies

            log_p_x = torch.cat(log_p_x)

            log_p.append((torch.logsumexp(log_p_x, 0) - np.log(len(log_p_x))).item())

        return np.mean(log_p)
