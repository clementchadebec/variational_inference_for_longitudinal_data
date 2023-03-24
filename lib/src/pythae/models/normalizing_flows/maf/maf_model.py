import os

import torch
import torch.nn as nn

from ...base.base_utils import ModelOutput
from ..base import BaseNF
from ..layers import BatchNorm
from ..made import MADE, MADEConfig
from .maf_config import MAFConfig


class MAF(BaseNF):
    """Masked Autoregressive Flow.

    Args:
        model_config (MAFConfig): The MAF model configuration setting the main parameters of the
            model.
    """

    def __init__(self, model_config: MAFConfig):

        BaseNF.__init__(self, model_config=model_config)

        self.net = []
        self.m = {}
        self.model_config = model_config
        self.hidden_size = model_config.hidden_size
        self.context_dim = model_config.context_dim

        self.model_name = "MAF"

        made_config = MADEConfig(
            input_dim=(self.input_dim,),
            output_dim=(self.input_dim,),
            hidden_sizes=[self.hidden_size] * self.model_config.n_hidden_in_made,
            degrees_ordering="sequential",
            context_dim=(self.context_dim,) if self.context_dim is not None else None,
        )

        for i in range(model_config.n_blocks):
            self.net.extend([MADE(made_config)])
            if self.model_config.include_batch_norm:
                self.net.extend([BatchNorm(self.input_dim)])

        self.net = nn.ModuleList(self.net)

    def forward(self, x: torch.Tensor, h: torch.Tensor = None, **kwargs) -> ModelOutput:
        """The input data is transformed toward the prior

        Args:
            x (torch.Tensor): An input tensor.
            h (torch.Tensor): The context tensor. Default: None.

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """

        x = x.reshape(x.shape[0], -1)
        sum_log_abs_det_jac = torch.zeros(x.shape[0]).to(x.device)

        for layer in self.net:
            if layer.__class__.__name__ == "MADE":
                layer_out = layer(x, h)
                mu, log_var = layer_out.mu, layer_out.log_var

                x = (x - mu) * (-log_var).exp()
                sum_log_abs_det_jac += -log_var.sum(dim=-1)  # - alpha

            else:
                layer_out = layer(x)
                x = layer_out.out
                sum_log_abs_det_jac += layer_out.log_abs_det_jac

            x = x.flip(dims=(1,))

        return ModelOutput(out=x, log_abs_det_jac=sum_log_abs_det_jac)

    def inverse(self, y: torch.Tensor, h: torch.Tensor = None, **kwargs) -> ModelOutput:
        """The prior is transformed toward the input data

        Args:
            x (torch.Tensor): An input tensor.
            h (torch.Tensor): The context tensor. Default: None.

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """

        y = y.reshape(y.shape[0], -1)
        sum_log_abs_det_jac = torch.zeros(y.shape[0]).to(y.device)

        for layer in self.net[::-1]:
            y = y.flip(dims=(1,))
            if layer.__class__.__name__ == "MADE":
                x = torch.zeros_like(y)
                for i in range(self.input_dim):
                    layer_out = layer(x.clone(), h=h)

                    mu, log_var = layer_out.mu, layer_out.log_var

                    x[:, i] = y[:, i] * (log_var[:, i]).exp() + mu[:, i]

                    sum_log_abs_det_jac += log_var[:, i]

                y = x
            else:
                layer_out = layer.inverse(y)
                y = layer_out.out
                sum_log_abs_det_jac += layer_out.log_abs_det_jac

        return ModelOutput(out=y, log_abs_det_jac=sum_log_abs_det_jac)
