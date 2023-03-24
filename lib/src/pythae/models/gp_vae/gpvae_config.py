from dataclasses import field
from typing import List, Union

from pydantic.dataclasses import dataclass
from typing_extensions import Literal

from ..vae import VAEConfig


@dataclass
class GPVAEConfig(VAEConfig):
    """GPVAE model config class.

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        kernel_choice (str): The kernel to choose. Available options are ['rbf', 'imq'] i.e.
            radial basis functions or inverse multiquadratic kernel. Default: 'imq'.
        reg_weight (float): The weight to apply between reconstruction and Maximum Mean
            Discrepancy. Default: 3e-2
        kernel_bandwidth (float): The kernel bandwidth. Default: 1
        scales (list): The scales to apply if using multi-scale imq kernels. If None, use a unique
            imq kernel. Default: [.1, .2, .5, 1., 2., 5, 10.].
        reconstruction_loss_scale (float): Parameter scaling the reconstruction loss. Default: 1
    """
    time_length: int = 1
    kernel_choice: Literal["cauchy", "rbf", "diffusion", "matern"] = "cauchy"
    kernel_scales: int = 1
    length_scale: float=1.
    sigma: float = 1.
    beta: float = 1.
    out_channels_time_cnn: int = 256
