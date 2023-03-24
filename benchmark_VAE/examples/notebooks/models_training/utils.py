import torch
from pythae.data.datasets import DatasetOutput
from torch.utils.data import Dataset
from pythae.models.nn import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput
import torch.nn as nn
import numpy as np


def make_batched_masks(data, prob_missing_data, batch_size):
    mask = torch.zeros(data.shape[:2], requires_grad=False)
    prob = ((1 - prob_missing_data) - 2 / data.shape[1]) * data.shape[1] / (data.shape[1] - 2)

    for i in range(int(data.shape[0] / batch_size)):

        bern = torch.distributions.Bernoulli(probs=prob).sample((data.shape[1]-2,))
        
        _mask = torch.zeros(data.shape[1])
        _mask[:2] = 1
        _mask[2:] = bern

        idx = np.random.rand(*_mask.shape).argsort(axis=-1)
        
        _mask = np.take_along_axis(_mask, idx, axis=-1)
        mask[i*batch_size:(i+1)*batch_size] = _mask.repeat(batch_size, 1)

    if data.shape[0] % batch_size > 0:

        bern = torch.distributions.Bernoulli(probs=prob).sample((data.shape[1]-2,))
        
        _mask = torch.zeros(data.shape[1])
        _mask[:2] = 1
        _mask[2:] = bern

        idx = np.random.rand(*_mask.shape).argsort(axis=-1)
        _mask = np.take_along_axis(_mask, idx, axis=-1)

        mask[-(data.shape[0] % batch_size):] = _mask.repeat((data.shape[0] % batch_size), 1)


    return mask


class My_MaskedDataset(Dataset):
    def __init__(self, data, seq_mask, pix_mask):
        self.data = data.type(torch.float)
        self.sequence_mask = seq_mask.type(torch.float)
        self.pixel_mask = pix_mask.type(torch.float)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        seq_m = self.sequence_mask[index]
        pix_m = self.pixel_mask[index] 
        #x = (x > torch.distributions.Uniform(0, 1).sample(x.shape).to(x.device)).float()
        return DatasetOutput(data=x, seq_mask=seq_m, pix_mask=pix_m)


class My_Dataset(Dataset):
    def __init__(self, data):
        self.data = data.type(torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        #x = (x > torch.distributions.Uniform(0, 1).sample(x.shape).to(x.device)).float()
        return DatasetOutput(data=x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        nn.Module.__init__(self)

        self.conv_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class EncoderPhysio(BaseEncoder):
    def __init__(self, args: dict):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        if hasattr(args, "context_dim"):
            self.context_dim = args.context_dim
        else:
            self.context_dim = None

        self.fc = nn.Sequential(
            nn.Linear(np.prod(args.input_dim), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.embedding = nn.Linear(256, self.latent_dim)
        self.log_var = nn.Linear(256, self.latent_dim)
        if self.context_dim is not None:
            self.context = nn.Linear(256, self.context_dim)

    def forward(self, x):
        output = ModelOutput()

        out = self.fc(x.reshape(-1, np.prod(self.input_dim)))

        output["embedding"] = self.embedding(out)
        output["log_covariance"] = self.log_var(out)
        if self.context_dim is not None:
            output["context"] = self.context(out)

        return output

### Define paper decoder network
class DecoderPhysio(BaseDecoder):
    def __init__(self, args: dict):
        BaseDecoder.__init__(self)

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim

        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, np.prod(self.input_dim)),
        )


    def forward(self, z: torch.Tensor):

        output = ModelOutput()

        out = self.fc(z)

        output["reconstruction"] = out.reshape((z.shape[0],) + self.input_dim)

        return output

### Define paper encoder network
class Encoder_ColorMNIST(BaseEncoder):
    def __init__(self, args: dict):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        if hasattr(args, "context_dim"):
            self.context_dim = args.context_dim
        else:
            self.context_dim = None

        self.fc = nn.Sequential(
            nn.Linear(np.prod(args.input_dim), 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
        )

        self.embedding = nn.Linear(256, self.latent_dim)
        self.log_var = nn.Linear(256, self.latent_dim)
        if self.context_dim is not None:
            self.context = nn.Linear(256, self.context_dim)

    def forward(self, x):
        output = ModelOutput()

        out = self.fc(x.reshape(-1, np.prod(self.input_dim)))

        output["embedding"] = self.embedding(out)
        output["log_covariance"] = self.log_var(out)
        if self.context_dim is not None:
            output["context"] = self.context(out)

        return output

class Encoder_ColorMNIST_GPVAE(BaseEncoder):
    def __init__(self, args: dict):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        if hasattr(args, "context_dim"):
            self.context_dim = args.context_dim
        else:
            self.context_dim = None

        self.fc = nn.Sequential(
            nn.Linear(np.prod(args.input_dim), 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
        )

        self.embedding = nn.Linear(256, self.latent_dim)
        self.log_var = nn.Linear(256, 2 * self.latent_dim)
        if self.context_dim is not None:
            self.context = nn.Linear(256, self.context_dim)

    def forward(self, x):
        output = ModelOutput()

        out = self.fc(x.reshape(-1, np.prod(self.input_dim)))

        output["embedding"] = self.embedding(out)
        output["log_covariance"] = self.log_var(out)
        if self.context_dim is not None:
            output["context"] = self.context(out)

        return output

### Define paper decoder network
class Decoder_ColorMNIST(BaseDecoder):
    def __init__(self, args: dict):
        BaseDecoder.__init__(self)

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim

        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, np.prod(self.input_dim)),
            nn.Sigmoid(),
        )


    def forward(self, z: torch.Tensor):

        output = ModelOutput()

        out = self.fc(z)

        output["reconstruction"] = out.reshape((z.shape[0],) + self.input_dim)

        return output


class Encoder_Chairs(BaseEncoder):

    def __init__(self, args: dict):
        BaseEncoder.__init__(self)

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        if hasattr(args, "context_dim"):
            self.context_dim = args.context_dim
        else:
            self.context_dim = None
        self.n_channels = args.input_dim[0]

        layers = nn.Sequential(
            nn.Conv2d(self.n_channels, 16, 4, 2, padding=1),
            nn.Conv2d(16, 32, 4, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2, padding=1),
            nn.LeakyReLU(),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),

        )

        self.layers = layers

        self.embedding = nn.Linear(128 * 4 * 4, args.latent_dim)
        self.log_var = nn.Linear(128 * 4 * 4, args.latent_dim)
        if self.context_dim is not None:
            self.context = nn.Linear(128 * 4 * 4, self.context_dim)

    def forward(self, x: torch.Tensor):
        output = ModelOutput()

        out = self.layers(x.reshape((-1,) + self.input_dim))
        output["embedding"] = self.embedding(out.reshape(-1, 128*4*4))
        output["log_covariance"] = self.log_var(out.reshape(-1, 128*4*4))
        if self.context_dim is not None:
            output["context"] = self.context(out)

        return output


class Encoder_Chairs_GPVAE(BaseEncoder):

    def __init__(self, args: dict):
        BaseEncoder.__init__(self)

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        if hasattr(args, "context_dim"):
            self.context_dim = args.context_dim
        else:
            self.context_dim = None
        self.n_channels = args.input_dim[0]
        

        layers = nn.Sequential(
            nn.Conv2d(self.n_channels, 16, 4, 2, padding=1),
            nn.Conv2d(16, 32, 4, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2, padding=1),
            nn.LeakyReLU(),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),

        )

        self.layers = layers

        self.embedding = nn.Linear(128 * 4 * 4, args.latent_dim)
        self.log_var = nn.Linear(128 * 4 * 4, 2 * args.latent_dim)
        if self.context_dim is not None:
            self.context = nn.Linear(128 * 4 * 4, self.context_dim)

    def forward(self, x: torch.Tensor):
        output = ModelOutput()

        out = self.layers(x.reshape((-1,) + self.input_dim))
        output["embedding"] = self.embedding(out.reshape(-1, 128*4*4))
        output["log_covariance"] = self.log_var(out.reshape(-1, 128*4*4))
        if self.context_dim is not None:
            output["context"] = self.context(out)

        return output


class Decoder_Chairs(BaseDecoder):
    def __init__(self, args: dict):
        BaseDecoder.__init__(self)

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.n_channels = args.input_dim[0]

        self.fc = nn.Linear(args.latent_dim, 128 * 4 * 4)

        layers = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, 2, padding=1),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
            nn.ConvTranspose2d(128, 64, 5, 2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 5, 2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, self.n_channels, 4, 2, padding=1)
            #nn.Sigmoid()
        )   

        self.layers = layers

    def forward(self, z: torch.Tensor):
        output = ModelOutput()

        out = self.fc(z).reshape(z.shape[0], 128, 4, 4)
        out = self.layers(out)

        output["reconstruction"] = out.reshape((z.shape[0],) + self.input_dim)

        return output

### Define paper encoder network
class Encoder_HMNIST(BaseEncoder):
    def __init__(self, args: dict):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        if hasattr(args, "context_dim"):
            self.context_dim = args.context_dim
        else:
            self.context_dim = None

        #self.conv = nn.Sequential(
        #    nn.Conv2d(1, 256, 3, padding=1),
        #    nn.ReLU(),
        #    nn.Conv2d(256, 1, 3, padding=1),
        #    nn.ReLU()
        #)

        #self.time_cnn = nn.Conv1d(np.prod(args.input_dim), args.out_channels_time_cnn, kernel_size=3, padding=1)

        self.fc = nn.Sequential(
            nn.Linear(np.prod(self.input_dim), 256),
            nn.ReLU(),
            #nn.Linear(256, 256),
            #nn.ReLU(),
        )

        self.embedding = nn.Linear(256, self.latent_dim)
        self.log_var = nn.Linear(256, 2*self.latent_dim)
        if self.context_dim is not None:
            self.context = nn.Linear(256, self.context_dim)

    def forward(self, x):
        output = ModelOutput()

        #x = torch.transpose(self.time_cnn(torch.transpose(x.reshape(x.shape[0], x.shape[1], -1), 2, 1)), 2, 1)
        #out = self.conv(x.reshape((-1,) + self.input_dim))
        out = self.fc(x.reshape(-1, np.prod(self.input_dim)))

        output["embedding"] = self.embedding(out)
        output["log_covariance"] = self.log_var(out)
        if self.context_dim is not None:
            output["context"] = self.context(out)

        return output

### Define paper decoder network
class Decoder_HMNIST(BaseDecoder):
    def __init__(self, args: dict):
        BaseDecoder.__init__(self)

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim

        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, np.prod(self.input_dim)),
            nn.Sigmoid()
        )


    def forward(self, z: torch.Tensor):

        output = ModelOutput()

        out = self.fc(z)

        output["reconstruction"] = out.reshape((z.shape[0],) + self.input_dim)

        return output


### Define paper encoder network
class Encoder_Sprites_Missing(BaseEncoder):
    def __init__(self, args: dict):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        if hasattr(args, "context_dim"):
            self.context_dim = args.context_dim
        else:
            self.context_dim = None

        self.conv = nn.Sequential(
            nn.Conv2d(3, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 3, 3, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(np.prod(args.input_dim), 32),
            nn.ReLU(),
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.embedding = nn.Linear(256, self.latent_dim)
        self.log_var = nn.Linear(256, self.latent_dim)
        if self.context_dim is not None:
            self.context = nn.Linear(256, self.context_dim)

    def forward(self, x):
        output = ModelOutput()
        #out = self.conv(x.reshape((-1,) + self.input_dim))
        out = self.fc(x.reshape(-1, np.prod(self.input_dim)))

        output["embedding"] = self.embedding(out)
        output["log_covariance"] = self.log_var(out)
        if self.context_dim is not None:
            output["context"] = self.context(out)

        return output

### Define paper decoder network
class Decoder_Sprites_Missing(BaseDecoder):
    def __init__(self, args: dict):
        BaseDecoder.__init__(self)

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim

        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, np.prod(self.input_dim)),
            nn.Sigmoid()
        )


    def forward(self, z: torch.Tensor):

        output = ModelOutput()

        out = self.fc(z)

        output["reconstruction"] = out.reshape((z.shape[0],) + self.input_dim)

        return output

class Encoder_Faces(BaseEncoder):

    def __init__(self, args: dict):
        BaseEncoder.__init__(self)

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        if hasattr(args, "context_dim"):
            self.context_dim = args.context_dim
        else:
            self.context_dim = None
        self.n_channels = args.input_dim[0]

        layers = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2),
            nn.Conv2d(16, 32, 5, 2),
            nn.Conv2d(32, 64, 5, 2),
            nn.Conv2d(64, 128, 5, 2),
            nn.Conv2d(128, 128, 5, 2),
            nn.Conv2d(128, 128, 5, 2),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
        )

        self.layers = layers

        self.embedding = nn.Linear(128 * 13 * 7, args.latent_dim)
        self.log_var = nn.Linear(128 * 13 * 7, args.latent_dim)
        if self.context_dim is not None:
            self.context = nn.Linear(128 * 13 * 7, self.context_dim)

    def forward(self, x: torch.Tensor):
        output = ModelOutput()

        out = self.layers(x.reshape((-1,) + self.input_dim))
        output["embedding"] = self.embedding(out.reshape(-1, 128*13*7))
        output["log_covariance"] = self.log_var(out.reshape(-1, 128*13*7))
        if self.context_dim is not None:
            output["context"] = self.context(out)

        return output

class Encoder_Faces_GPVAE(BaseEncoder):

    def __init__(self, args: dict):
        BaseEncoder.__init__(self)

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        if hasattr(args, "context_dim"):
            self.context_dim = args.context_dim
        else:
            self.context_dim = None
        self.n_channels = args.input_dim[0]
        

        layers = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2),
            nn.Conv2d(16, 32, 5, 2),
            nn.Conv2d(32, 64, 5, 2),
            nn.Conv2d(64, 128, 5, 2),
            nn.Conv2d(128, 128, 5, 2),
            nn.Conv2d(128, 128, 5, 2),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
        )

        self.layers = layers

        self.embedding = nn.Linear(128 * 13 * 7, args.latent_dim)
        self.log_var = nn.Linear(128 * 13 * 7, 2 * args.latent_dim)
        if self.context_dim is not None:
            self.context = nn.Linear(128 * 13 * 7, self.context_dim)

    def forward(self, x: torch.Tensor):
        output = ModelOutput()

        out = self.layers(x.reshape((-1,) + self.input_dim))
        output["embedding"] = self.embedding(out.reshape(-1, 128*4*4))
        output["log_covariance"] = self.log_var(out.reshape(-1, 128*4*4))
        if self.context_dim is not None:
            output["context"] = self.context(out)

        return output


class Decoder_Faces(BaseDecoder):
    def __init__(self, args: dict):
        BaseDecoder.__init__(self)

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.n_channels = args.input_dim[0]

        self.fc = nn.Linear(args.latent_dim, 128 * 13 * 7)

        layers = nn.Sequential(
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
            nn.ConvTranspose2d(128, 128, 5, 2, output_padding=(0, 1)),
            nn.ConvTranspose2d(128, 128, 5, 2, output_padding=(0, 1)),
            nn.ConvTranspose2d(128, 64, 5, 2, padding=(0, 0)),
            nn.ConvTranspose2d(64, 32, 5, 2, padding=(0, 0), output_padding=(1, 0)),
            nn.ConvTranspose2d(32, 16, 5, 2, padding=(0, 1), output_padding=(0, 1)),
            nn.ConvTranspose2d(16, 3, (4, 5), 2, padding=(0, 1)), 
        )

        self.layers = layers

    def forward(self, z: torch.Tensor):
        output = ModelOutput()

        out = self.fc(z).reshape(z.shape[0], 128, 13, 7)
        out = self.layers(out)

        output["reconstruction"] = out.reshape((z.shape[0],) + self.input_dim)

        return output