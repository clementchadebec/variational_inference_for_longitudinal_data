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
from .liwae_iaf_config import LIWAE_IAF_Config


class LIWAE_IAF(VAE):
    """Longitudinal Importance Weighted Auto Encoder with Inverse Autoregressive Flows
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
        model_config: LIWAE_IAF_Config,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        self.model_name = "LIWAE_IAF"

        self.n_obs = model_config.n_obs_per_ind
        self.warmup = model_config.warmup
        self.context_dim = model_config.context_dim
        self.n_samples = model_config.number_samples
        self.prior = "standard"
        self.posterior = "gaussian"
       
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
        epoch = kwargs.pop("epoch", 100)

        if epoch < self.warmup:
            encoder_output = self.encoder(x)#, torch.arange(0, self.n_obs).to(x.device).repeat(x.shape[0]).unsqueeze(-1) / self.n_obs)
            mu, log_var = encoder_output.embedding, encoder_output.log_covariance

            mu = mu.unsqueeze(1).repeat(1, self.n_samples, 1)
            log_var = log_var.unsqueeze(1).repeat(1, self.n_samples, 1)

            std = torch.exp(0.5 * log_var)
            z, _ = self._sample_gauss(mu, std)

            z_seq = z

            recon_x = self.decoder(z_seq.reshape(-1, self.latent_dim))["reconstruction"].reshape(x.shape[0]*self.n_obs, self.n_samples, -1)#, torch.arange(0, self.n_obs).to(x.device).repeat(x.shape[0]).unsqueeze(-1) / self.n_obs)["reconstruction"] # [B*n_obs x input_dim]

            loss, recon_loss, kld = self._iwae_loss_function(
                recon_x=recon_x,
                x=x.reshape((x.shape[0]*self.n_obs,) + x.shape[2:]),
                mu=mu,
                log_var=log_var,
                z=z
            )

        else:

            # if missing data pick randomly index of non missing data
            vi_index = np.random.randint(self.n_obs)

            encoder_output = self.encoder(x[:, vi_index])#, vi_index * torch.ones(x.shape[0], 1).to(x.device) / self.n_obs)

            mu, log_var = encoder_output.embedding, encoder_output.log_covariance
            h = None#encoder_output.context

            mu = mu.unsqueeze(1).repeat(1, self.n_samples, 1)
            log_var = log_var.unsqueeze(1).repeat(1, self.n_samples, 1)

            std = torch.exp(0.5 * log_var)
            z, _ = self._sample_gauss(mu, std)

            z_vi_index = z.reshape(-1, self.latent_dim)

            ## propagate in past
            z_seq = []
            z_rev = z_vi_index
            log_abs_det_jac = 0
            for i in range(vi_index - 1, -1, -1):
                #print("past", i)
                flow_output = self.flows[i](z_rev)
                z_rev = flow_output.out
                log_abs_det_jac += flow_output.log_abs_det_jac
                z_seq.append(z_rev)
##
            z_seq.reverse()
#
            z_seq.append(z_vi_index)

            #proapagate in future
            z_for = z_vi_index
            for i in range(vi_index, self.n_obs - 1):
                #print("future", i)
                flow_output = self.flows[i].inverse(z_for)
                z_for = flow_output.out
                z_seq.append(z_for)

            z_seq = torch.cat(z_seq, dim=-1)

            #t = torch.linspace(0, 1, self.n_obs).repeat(x.shape[0], 1).to(z.device)
            #z = torch.cat((t.unsqueeze(-1), z_seq.reshape(x.shape[0], -1, self.latent_dim)), dim=-1)

            recon_x = self.decoder(z_seq.reshape(-1, self.latent_dim))["reconstruction"].reshape(x.shape[0]*self.n_obs, self.n_samples, -1)#, torch.arange(0, self.n_obs).to(x.device).repeat(x.shape[0]).unsqueeze(-1) / self.n_obs)["reconstruction"] # [B*n_obs x input_dim]

            z_seq = z_seq.reshape(x.shape[0], self.n_obs, self.n_samples, self.latent_dim)

            loss, recon_loss, kld = self.loss_function(
                recon_x=recon_x,
                x=x,
                mu=mu,
                log_var=log_var,
                z_seq=z_seq,
                z_0_vi_index=z_vi_index,
                log_abs_det_jac=log_abs_det_jac,
                epoch=epoch,
            )

        output = ModelOutput(
            reconstruction_loss=recon_loss,
            reg_loss=kld,
            loss=loss,
            recon_x=recon_x.reshape(x.shape[0]*self.n_obs, self.n_samples, -1)[:, 0, :].reshape_as(x),
            z=z,
            z_seq=z_seq,
            x=x
        )

        return output


    def _iwae_loss_function(self, recon_x, x, mu, log_var, z):

        if self.model_config.reconstruction_loss == "mse":

            recon_loss = F.mse_loss(
                recon_x,
                x.reshape(recon_x.shape[0], -1)
                .unsqueeze(1)
                .repeat(1, self.n_samples, 1),
                reduction="none",
            ).sum(dim=-1)

        elif self.model_config.reconstruction_loss == "bce":

            recon_loss = F.binary_cross_entropy(
                recon_x,
                x.reshape(recon_x.shape[0], -1)
                .unsqueeze(1)
                .repeat(1, self.n_samples, 1),
                reduction="none",
            ).sum(dim=-1)

        log_q_z = (-0.5 * (log_var + torch.pow(z - mu, 2) / log_var.exp())).sum(dim=-1)
        log_p_z = -0.5 * (z ** 2).sum(dim=-1)

        KLD = -(log_p_z - log_q_z)

        log_w = -(recon_loss + KLD)

        log_w_minus_max = log_w - log_w.max(1, keepdim=True)[0]
        w = log_w_minus_max.exp()
        w_tilde = (w / w.sum(axis=1, keepdim=True)).detach()

        return (
            -(w_tilde * log_w).sum(1).mean(dim=0),
            recon_loss.mean(dim=0),
            KLD.mean(dim=0),
        )

    def loss_function(self, recon_x, x, mu, log_var, z_0_vi_index, z_seq, log_abs_det_jac, epoch):
        if self.model_config.reconstruction_loss == "mse":
            recon_loss = 0.5 * (
                    F.mse_loss(
                        recon_x,
                        x.reshape(recon_x.shape[0], -1).unsqueeze(1).repeat(1, self.n_samples, 1),
                        reduction="none"
                    )
                ).sum(dim=-1).reshape(x.shape[0], self.n_obs, -1).mean(dim=1)
                
        elif self.model_config.reconstruction_loss == "bce":

            recon_loss = (
                    F.binary_cross_entropy(
                        recon_x,
                        x.reshape(recon_x.shape[0], -1).unsqueeze(1).repeat(1, self.n_samples, 1),
                        reduction="none"
                    )
                ).sum(dim=-1).reshape(x.shape[0], self.n_obs, -1).mean(dim=1)
                
        z0 = z_seq[:, 0]

        # starting gaussian log-density
        log_prob_z_vi_index = (
            -0.5 * (log_var + torch.pow(z_0_vi_index.reshape_as(mu) - mu, 2) / torch.exp(log_var))
        ).sum(dim=-1)

        log_p_z = self._log_p_z(z0) 

        # prior log-density
        log_prior_z_vi_index = (log_p_z.reshape(-1) + torch.clamp(torch.tensor(log_abs_det_jac), min=1e-8, max=1e8)).reshape(-1, self.n_samples)

        KLD = log_prob_z_vi_index - log_prior_z_vi_index

        log_w = -(recon_loss + KLD) + 1e-8

        log_w_minus_max = log_w - log_w.max(1, keepdim=True)[0]
        w = log_w_minus_max.exp()
        w_tilde = (w / w.sum(axis=1, keepdim=True)).detach()

        if -(w_tilde * log_w).sum(1).mean() != -(w_tilde * log_w).sum(1).mean():
            print("sum", w.sum(axis=1, keepdim=True))
            print("w", w)
            print("log_w", log_w)
            print("log_jac", log_abs_det_jac)
            print("KLD", KLD)
            print("prior", log_p_z)

        return (
            -(w_tilde * log_w).sum(1).mean(),
            recon_loss.mean(),
            KLD.mean(),
        )
    def _log_p_z(self, z):
        #if self.prior == "standard":
        log_p_z = (-0.5 * torch.pow(z, 2)).sum(dim=-1)

        return log_p_z


    def generate(self, z):
        z_for = z

        #_rec = self.decoder(z).reconstruction
        #h = self.encoder(_rec).context

        z_seq = [z]
        for i in range(self.n_obs - 1):
            flow_output = self.flows[i].inverse(z_for)
            z_for = flow_output.out
            z_seq.append(z_for)

        z_seq = torch.cat(z_seq, dim=-1)

        #t = torch.linspace(0, 1, self.n_obs).repeat(z.shape[0], 1).to(z.device)
        #z = torch.cat((t.unsqueeze(-1), z_seq.reshape(z.shape[0], -1, self.latent_dim)), dim=-1)
        return self.decoder(z_seq.reshape(-1, self.latent_dim))["reconstruction"].reshape((z.shape[0], self.n_obs,) + self.input_dim)#, torch.arange(0, self.n_obs).to(z.device).repeat(z.shape[0]).unsqueeze(-1) / self.n_obs)["reconstruction"].reshape((z.shape[0], self.n_obs,) + self.input_dim)

    def infer_missing(self, x, seq_mask, pix_mask):
        # iterate on seen images in sequence and keep the one maximizing p(x_i^obs|z)
        
        p_x_given_z = []
        reconstructions = []
        x = x * pix_mask * seq_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        
        for vi_index in range(self.n_obs):

            if seq_mask[0][vi_index] != 0:

                encoder_output = self.encoder(x[:, vi_index])#, vi_index * torch.ones(x.shape[0], 1).to(x.device) / self.n_obs)

                mu, log_var = encoder_output.embedding, encoder_output.log_covariance
                h = None#encoder_output.context

                std = torch.exp(0.5 * log_var)
                z, _ = self._sample_gauss(mu, std)

                z_0_vi_index = z

                log_abs_det_jac_posterior = 0
                if self.posterior == 'iaf':

                    if self.posterior_iaf_config.context_dim is not None:
                        try:
                            h = encoder_output.context

                        except AttributeError as e:
                            raise AttributeError(
                                "Cannot get context from encoder outputs. If you set `context_dim` argument to "
                                "something different from None please ensure that the encoder actually outputs "
                                f"the context vector 'h'. Exception caught: {e}."
                            )

                        # Pass it through the Normalizing flows
                        flow_output = self.posterior_iaf_flow.inverse(z, h=h)  # sampling

                    else:
                        # Pass it through the Normalizing flows
                        flow_output = self.posterior_iaf_flow.inverse(z)  # sampling

                    z = flow_output.out
                    log_abs_det_jac_posterior += flow_output.log_abs_det_jac

                z_vi_index = z

                ## propagate in past
                z_seq = []
                z_rev = z_vi_index
                log_abs_det_jac = 0
                for i in range(vi_index - 1, -1, -1):
                    #print("past", i)
                    flow_output = self.flows[i](z_rev)
                    z_rev = flow_output.out
                    log_abs_det_jac += flow_output.log_abs_det_jac
                    z_seq.append(z_rev)

                z_seq.reverse()

                z_seq.append(z_vi_index)

                #proapagate in future
                z_for = z_vi_index
                for i in range(vi_index, self.n_obs - 1):
                    #print("future", i)
                    flow_output = self.flows[i].inverse(z_for)
                    z_for = flow_output.out
                    z_seq.append(z_for)

                z_seq = torch.cat(z_seq, dim=-1)

                #t = torch.linspace(0, 1, self.n_obs).repeat(x.shape[0], 1).to(z.device)
                #z = torch.cat((t.unsqueeze(-1), z_seq.reshape(x.shape[0], -1, self.latent_dim)), dim=-1)

                recon_x = self.decoder(z_seq.reshape(-1, self.latent_dim))["reconstruction"]#, torch.arange(0, self.n_obs).to(x.device).repeat(x.shape[0]).unsqueeze(-1) / self.n_obs)["reconstruction"] # [B*n_obs x input_dim]

                z_seq = z_seq.reshape(x.shape[0], self.n_obs, self.latent_dim)

                if self.model_config.reconstruction_loss == "mse":
                    recon_loss = (
                        0.5 * (
                            F.mse_loss(
                                recon_x.reshape(x.shape[0]*self.n_obs, -1),
                                x.reshape(x.shape[0]*self.n_obs, -1),
                                reduction="none"
                            ) * pix_mask.reshape(x.shape[0]*self.n_obs, -1)
                        ).sum(dim=-1).reshape(x.shape[0], -1) * seq_mask
                    ).mean(dim=-1)

                elif self.model_config.reconstruction_loss == "bce":
                
                    recon_loss = (
                        (
                            F.binary_cross_entropy(
                                recon_x.reshape(x.shape[0]*self.n_obs, -1),
                                x.reshape(x.shape[0]*self.n_obs, -1),
                                reduction="none"
                            ) * pix_mask.reshape(x.shape[0]*self.n_obs, -1)
                        ).sum(dim=-1).reshape(x.shape[0], -1) * seq_mask
                    ).mean(dim=-1)

                p_x_given_z.append(-recon_loss)
                reconstructions.append(recon_x)

        print(torch.cat(p_x_given_z).reshape(-1, x.shape[0])[0])
        idx = torch.argsort(torch.cat(p_x_given_z).reshape(-1, x.shape[0]), dim=0, descending=True)
        print(idx, torch.cat(reconstructions).reshape((-1, x.shape[0], self.n_obs,)+x.shape[2:]).shape)
        #idx = [int(i + self.n_obs*k) for (k, i) in enumerate(idx)]
        #print(idx, torch.cat(reconstructions).reshape((-1, x.shape[0], self.n_obs,)+x.shape[2:])[idx].shape)
        return torch.cat(reconstructions).reshape((-1, x.shape[0], self.n_obs,)+x.shape[2:]), idx

            



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

                vi_index = np.random.randint(self.n_obs)

                x_rep = torch.cat(batch_size * [x])

                encoder_output = self.encoder(x_rep[:, vi_index])#, vi_index * torch.ones(x.shape[0], 1).to(x.device) / self.n_obs)

                mu, log_var = encoder_output.embedding, encoder_output.log_covariance
                h = None#encoder_output.context

                std = torch.exp(0.5 * log_var)
                z, _ = self._sample_gauss(mu, std)

                z_0_vi_index = z

                log_abs_det_jac_posterior = 0
                if self.posterior == 'iaf':

                    if self.posterior_iaf_config.context_dim is not None:
                        try:
                            h = encoder_output.context

                        except AttributeError as e:
                            raise AttributeError(
                                "Cannot get context from encoder outputs. If you set `context_dim` argument to "
                                "something different from None please ensure that the encoder actually outputs "
                                f"the context vector 'h'. Exception caught: {e}."
                            )

                        # Pass it through the Normalizing flows
                        flow_output = self.posterior_iaf_flow.inverse(z, h=h)  # sampling

                    else:
                        # Pass it through the Normalizing flows
                        flow_output = self.posterior_iaf_flow.inverse(z)  # sampling

                    z = flow_output.out
                    log_abs_det_jac_posterior += flow_output.log_abs_det_jac

                z_vi_index = z

                ## propagate in past
                z_seq = []
                z_rev = z_vi_index
                log_abs_det_jac = 0
                for j in range(vi_index - 1, -1, -1):
                    #print("past", i)
                    flow_output = self.flows[j](z_rev)
                    z_rev = flow_output.out
                    log_abs_det_jac += flow_output.log_abs_det_jac
                    z_seq.append(z_rev)
                z_seq.reverse()
                z_seq.append(z_vi_index)

                #proapagate in future
                z_for = z_vi_index
                for j in range(vi_index, self.n_obs - 1):
                    #print("future", i)
                    flow_output = self.flows[j].inverse(z_for)
                    z_for = flow_output.out
                    z_seq.append(z_for)

                z_seq = torch.cat(z_seq, dim=-1)

                recon_x = self.decoder(z_seq.reshape(-1, self.latent_dim))["reconstruction"]#, torch.arange(0, self.n_obs).to(x.device).repeat(x.shape[0]).unsqueeze(-1) / self.n_obs)["reconstruction"] # [B*n_obs x input_dim]

                z_seq = z_seq.reshape(x_rep.shape[0], self.n_obs, self.latent_dim)

                
                if self.model_config.reconstruction_loss == "mse":

                    log_p_x_given_z = (-0.5 * F.mse_loss(
                            recon_x.reshape(x_rep.shape[0]*self.n_obs, -1),
                            x_rep.reshape(x_rep.shape[0]*self.n_obs, -1),
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
                        recon_x.reshape(x_rep.shape[0]*self.n_obs, -1),
                        x_rep.reshape(x_rep.shape[0]*self.n_obs, -1),
                        reduction="none",
                    ).sum(dim=-1).reshape(x_rep.shape[0], -1).sum(dim=-1)

                z0 = z_seq[:, 0]

                # starting gaussian log-density
                log_prob_z_vi_index = (
                    -0.5 * (log_var + torch.pow(z_0_vi_index - mu, 2) / torch.exp(log_var))
                ).sum(dim=1) - log_abs_det_jac_posterior

                log_p_z = self._log_p_z(z0) 

                # prior log-density
                log_prior_z_vi_index = log_p_z + log_abs_det_jac

                log_p_x.append(
                    log_p_x_given_z + log_prior_z_vi_index - log_prob_z_vi_index
                )  # log(2*pi) simplifies

            log_p_x = torch.cat(log_p_x)

            log_p.append((torch.logsumexp(log_p_x, 0) - np.log(len(log_p_x))).item())

            if i % 100 == 0:
                print(f"Current nll at {i}: {np.mean(log_p)}")

        return np.mean(log_p)
