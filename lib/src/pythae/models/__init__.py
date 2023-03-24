""" 
This is the heart of pythae! 
Here are implemented some of the most common (Variational) Autoencoders models.

By convention, each implemented model is stored in a folder located in :class:`pythae.models`
and named likewise the model. The following modules can be found in this folder:

- | *modelname_config.py*: Contains a :class:`ModelNameConfig` instance inheriting
    from either :class:`~pythae.models.base.AEConfig` for Autoencoder models or 
    :class:`~pythae.models.base.VAEConfig` for Variational Autoencoder models. 
- | *modelname_model.py*: An implementation of the model inheriting either from
    :class:`~pythae.models.AE` for Autoencoder models or 
    :class:`~pythae.models.base.VAE` for Variational Autoencoder models. 
- *modelname_utils.py* (optional): A module where utils methods are stored.
"""

from .ae import AE, AEConfig
from .auto_model import AutoModel
from .base import BaseAE, BaseAEConfig
from .vae import VAE, VAEConfig
from .vamp import VAMP, VAMPConfig
from .lvae_iaf import LVAE_IAF, LVAE_IAF_Config
from .gp_vae import GPVAE, GPVAEConfig

__all__ = [
    "AutoModel",
    "BaseAE",
    "BaseAEConfig",
    "AE",
    "AEConfig",
    "VAE",
    "VAEConfig",
    "VAMP",
    "VAMPConfig",
    "LVAE_IAF",
    "LVAE_IAF_Config",
    "GPVAE",
    "GPVAEConfig"
]
