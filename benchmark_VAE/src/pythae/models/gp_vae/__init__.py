"""This module is the implementation of the Wasserstein Autoencoder proposed in 
(https://arxiv.org/abs/1711.01558).

Available samplers
-------------------

.. autosummary::
    ~pythae.samplers.NormalSampler
    ~pythae.samplers.GaussianMixtureSampler
    ~pythae.samplers.MAFSampler
    ~pythae.samplers.IAFSampler
    :nosignatures:
"""

from .gpvae_config import GPVAEConfig
from .gpvae_model import GPVAE

__all__ = ["GPVAE", "GPVAEConfig"]
