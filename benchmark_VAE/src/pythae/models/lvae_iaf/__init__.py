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

from .lvae_iaf_config import LVAE_IAF_Config
from .lvae_iaf_model import LVAE_IAF

__all__ = ["LVAE_IAF", "LVAE_IAF_Config"]
