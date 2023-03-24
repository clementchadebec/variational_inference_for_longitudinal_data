import os
import torch
import logging
from pythae.models import AutoModel
from pytorch_fid.fid_score import calculate_fid_given_arrays
import argparse
import numpy as np

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)

ap = argparse.ArgumentParser()

# Training setting
ap.add_argument(
    "--dataset",
    type=str,
    default="colormnist",
    choices=["colormnist", "3d_chairs", "sprites", "starmen", "rotated_mnist"],
    help="Path to dataset.",
    required=True,
)
ap.add_argument(
    "--model_path",
    type=str
)
ap.add_argument(
    "--use_wandb",
    help="whether to log the metrics in wandb",
    action="store_true",
)
ap.add_argument(
    "--wandb_project",
    help="wandb project name",
    default="longitudinal_vae_iaf",
)
ap.add_argument(
    "--wandb_entity",
    help="wandb entity name",
    default="your-name",
)

args = ap.parse_args()


def main(args):

    device = "cuda" if torch.cuda.is_available() else 'cpu'
    
    if args.dataset == "colormnist":
        test_data = torch.load(os.path.join('../../data/colormnist/test_12_long_color_mnist.pt'), map_location="cpu")

    elif args.dataset == "3d_chairs":
        test_data = torch.load(os.path.join('../../data/3d_chairs/3D_chairs_color.pt'), map_location="cpu")[1200:]

    elif args.dataset == "sprites":
        test_data = torch.load(os.path.join('../../data/sprites/Sprites_test.pt'), map_location="cpu")['data']

    elif args.dataset == "starmen":
        test_data = torch.load(os.path.join('../../data/starmen/starmen_1000.pt'), map_location="cpu")[900:].float()

    elif args.dataset == "rotated_mnist":
        test_data = torch.load(os.path.join('../../data/rotated_mnist/test_mnist_rotated_8.pt'), map_location="cpu")


    ### Reload model
    model = AutoModel.load_from_folder(os.path.join(args.model_path)).to(device).eval()

    n_gen_lvae = test_data.shape[0]
    n_gen_vae = test_data.shape[0] * test_data.shape[1]

    if model.model_name == "LVAE_IAF":
       
        batch_size = 500
        if model.prior == "vamp":
            batch_size = min(model.idle_input.shape[0], batch_size)
    
        full_batch_nbr = int(n_gen_lvae / batch_size)
        last_batch_samples_nbr = n_gen_lvae % batch_size

        gen_list = []

        for i in range(full_batch_nbr):

            if model.prior == "standard":
                    print("standard prior")
                    z = torch.randn(batch_size, model.latent_dim).cuda()
            elif model.prior == "vamp":
                print("vamp prior")
                means = model.pseudo_inputs(model.idle_input.to(device))[
                            :batch_size
                        ].reshape((batch_size,) + model.model_config.input_dim)

                encoder_output = model.encoder(means)
                mu, log_var = (
                    encoder_output.embedding,
                    encoder_output.log_covariance,
                )
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                z = mu + eps * std

            gen = model.generate(z).detach().cpu()
            gen_list.append(gen)

        if last_batch_samples_nbr > 0:

            if model.prior == "standard":
                    print("standard prior")
                    z = torch.randn(last_batch_samples_nbr, model.latent_dim).cuda()
            elif model.prior == "vamp":
                print("vamp prior")
                means = model.pseudo_inputs(model.idle_input.to(device))[
                            :batch_size
                        ].reshape((batch_size,) + model.model_config.input_dim)

                encoder_output = model.encoder(means)
                mu, log_var = (
                    encoder_output.embedding,
                    encoder_output.log_covariance,
                )
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                z = mu + eps * std

            gen = model.generate(z).detach().cpu()
            gen_list.append(gen)

        gen = torch.clamp(torch.cat(gen_list), 0, 1)


    elif model.model_name == "VAE":
        from pythae.samplers import NormalSampler
        sampler = NormalSampler(model=model)
        gen = torch.clamp(sampler.sample(num_samples=n_gen_vae, return_gen=True), 0, 1)

    elif model.model_name == "VAMP":
        from pythae.samplers import VAMPSampler
        sampler = VAMPSampler(model=model)
        gen = torch.clamp(sampler.sample(num_samples=n_gen_vae, return_gen=True), 0, 1)


    gen = (255. * gen).type(torch.uint8)
    test_data = (255. * test_data).type(torch.uint8)

    print("Generated shapes :", gen.shape)
    print("Test shapes :", test_data.shape)

    fid = calculate_fid_given_arrays(np.moveaxis(gen.cpu().numpy().reshape((-1,) + test_data.shape[2:]), 1, 3) / 255., np.moveaxis(test_data.cpu().numpy().reshape((-1,) + test_data.shape[2:]), 1, 3) / 255., batch_size=50, device='cuda:0', dims=2048)
    print("FID: ", fid)

if __name__ == "__main__":
    print(args)
    main(args)
