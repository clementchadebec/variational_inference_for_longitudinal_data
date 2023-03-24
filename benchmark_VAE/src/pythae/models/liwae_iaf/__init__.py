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

from .liwae_iaf_config import LIWAE_IAF_Config
from .liwae_iaf_model import LIWAE_IAF

__all__ = ["LIWAE_IAF", "LIWAE_IAF_Config"]
