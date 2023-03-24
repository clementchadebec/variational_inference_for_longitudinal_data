"""This module is the implementation of a Variational Autoencoder with Inverse Autoregressive Flow 
to enhance the expressiveness of the posterior distribution. 
(https://arxiv.org/abs/1606.04934).

Available samplers
-------------------

.. autosummary::
    ~pythae.samplers.NormalSampler
    ~pythae.samplers.GaussianMixtureSampler
    ~pythae.samplers.TwoStageVAESampler
    ~pythae.samplers.MAFSampler
    ~pythae.samplers.IAFSampler
    :nosignatures:
"""

from .lcvae_config import LCVAE_Config
from .lcvae_model import LCVAE

__all__ = ["LCVAE", "LCVAE_Config"]
