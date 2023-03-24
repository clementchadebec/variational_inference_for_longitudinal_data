import os

import torch
import logging
from pythae.models import VAE, VAMP, IWAE, VAEConfig, VAMPConfig, IWAEConfig, AutoModel
from pythae.trainers import BaseTrainerConfig, BaseTrainer
import argparse
import numpy as np

from utils import Encoder_ColorMNIST, Encoder_Chairs, Decoder_ColorMNIST, Decoder_Chairs, My_Dataset, My_MaskedDataset, make_batched_masks

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
    "--latent_dim",
    type=int,
    default=16
)
ap.add_argument(
    "--model",
    choices=[
        "vae",
        "vamp",
        "iwae"
    ],
    default="vae"
)
ap.add_argument(
    "--prob_missing_pixels",
    type=float,
    default=0.,
    help='The probability of missing pixels in the images'
)
ap.add_argument(
    "--vamp_number_components",
    type=int,
    default=500
)
ap.add_argument(
    "--iwae_number_samples",
    type=int,
    default=3
)
ap.add_argument(
    "--linear_scheduling_steps",
    type=int,
    default=10
)
ap.add_argument(
    "--num_epochs",
    type=int,
    default=100
)
ap.add_argument(
    "--learning_rate",
    type=float,
    default=1e-3
)
ap.add_argument(
    "--batch_size",
    type=int,
    default=128
)
ap.add_argument(
    "--steps_saving",
    type=int,
    default=None
)
ap.add_argument(
    "--steps_predict",
    type=int,
    default=3
)
ap.add_argument(
    "--compute_nll",
    action="store_true",
)
ap.add_argument(
    "--nll_n_samples",
    type=int,
    default=500,
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
    default="clementchadebec",
)

args = ap.parse_args()


def main(args):

    device = "cuda" if torch.cuda.is_available() else 'cpu'
    shuffle_data = True 

    if args.dataset == "colormnist":
        args.input_dim = (3, 28, 28)
        train_data = torch.load(os.path.join('/home/clement/Documents/TopographicVAE/tvae/data/train_12_long_color_mnist.pt'), map_location="cpu")#[:10000]
        eval_data = torch.load(os.path.join('/home/clement/Documents/TopographicVAE/tvae/data/val_12_long_color_mnist.pt'), map_location="cpu")#[:5000]


    elif args.dataset == "3d_chairs":
        args.input_dim = (3, 64, 64)
        train_data = torch.load(os.path.join('/gpfswork/rech/wlr/uhw48em/rvae/data/3d_chairs/3D_chairs_color.pt'), map_location="cpu")[:1000]
        eval_data = torch.load(os.path.join('/gpfswork/rech/wlr/uhw48em/rvae/data/3d_chairs/3D_chairs_color.pt'), map_location="cpu")[1000:1200]

    elif args.dataset == "sprites":
        args.input_dim = (3, 64, 64)
        train_data = torch.load(os.path.join('/gpfswork/rech/wlr/uhw48em/rvae/data/sprites/Sprites_train.pt'), map_location="cpu")['data'][:-1000]
        eval_data = torch.load(os.path.join('/gpfswork/rech/wlr/uhw48em/rvae/data/sprites/Sprites_train.pt'), map_location="cpu")['data'][-1000:]


    elif args.dataset == "starmen":
        args.input_dim = (1, 64, 64)
        train_data = torch.load(os.path.join('/gpfswork/rech/wlr/uhw48em/rvae/data/starmen/starmen_1000.pt'), map_location="cpu")[:700]
        eval_data = torch.load(os.path.join('/gpfswork/rech/wlr/uhw48em/rvae/data/starmen/starmen_1000.pt'), map_location="cpu")[700:900]

    elif args.dataset == "rotated_mnist":
        args.input_dim = (1, 28, 28)
        train_data = torch.load(os.path.join('/gpfswork/rech/wlr/uhw48em/rvae/data/rotated_mnist/train_mnist_rotated_8.pt'), map_location="cpu")[:-1000]
        eval_data = torch.load(os.path.join('/gpfswork/rech/wlr/uhw48em/rvae/data/rotated_mnist/train_mnist_rotated_8.pt'), map_location="cpu")[-1000:]

    train_seq_mask = torch.ones(train_data.shape[:2], requires_grad=False).type(torch.bool)
    eval_seq_mask = torch.ones(eval_data.shape[:2], requires_grad=False).type(torch.bool)

    if args.prob_missing_pixels > 0.:
        train_pix_mask = torch.distributions.Bernoulli(probs=1-args.prob_missing_pixels).sample((train_data.shape[0], train_data.shape[1],)+train_data.shape[-2:]).unsqueeze(2).repeat(1, 1, train_data.shape[2], 1, 1)
        eval_pix_mask = torch.distributions.Bernoulli(probs=1-args.prob_missing_pixels).sample((eval_data.shape[0], eval_data.shape[1],)+eval_data.shape[-2:]).unsqueeze(2).repeat(1, 1, train_data.shape[2], 1, 1)

    else:
        train_pix_mask = torch.ones_like(train_data, requires_grad=False).type(torch.bool)
        eval_pix_mask = torch.ones_like(eval_data, requires_grad=False).type(torch.bool)

    if args.model =="vae":
        config = VAEConfig(
            input_dim=args.input_dim,
            latent_dim=args.latent_dim
        )
    elif args.model == "vamp":
        config = VAMPConfig(
            input_dim=args.input_dim,
            latent_dim=args.latent_dim,
            number_components=args.vamp_number_components,
            linear_scheduling_steps=args.linear_scheduling_steps
        )
    elif args.model == "iwae":
        config = IWAEConfig(
            input_dim=args.input_dim,
            latent_dim=args.latent_dim,
            number_samples=args.iwae_number_samples
        )

    training_config = BaseTrainerConfig(
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        steps_saving=args.steps_saving,
        steps_predict=args.steps_predict,
        shuffle=shuffle_data
    )

    if args.dataset == "colormnist":

        encoder = Encoder_ColorMNIST(config)
        decoder = Decoder_ColorMNIST(config)

        if args.model == "vae":
            model = VAE(config, encoder, decoder).cuda()
        elif args.model == "vamp":
            model = VAMP(config, encoder, decoder).cuda()
        elif args.model == "iwae":
            model = IWAE(config, encoder, decoder).cuda()

        ### Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate, eps=1e-4)

        ### Scheduler
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[50, 100, 150, 200],
            gamma=10**(-1/4),
            verbose=True
        )

    elif args.dataset == "3d_chairs":

        encoder = Encoder_Chairs(config)
        decoder = Decoder_Chairs(config)

        if args.model == "vae":
            model = VAE(config, encoder, decoder).cuda()
        elif args.model == "vamp":
            model = VAMP(config, encoder, decoder).cuda()
        elif args.model == "iwae":
            model = IWAE(config, encoder, decoder).cuda()

        ### Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate, eps=1e-4)

        ### Scheduler
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[150, 200, 250, 300, 350],
            gamma=0.5,
            verbose=True
        )

    elif args.dataset == "sprites":

        encoder = Encoder_Chairs(config)
        decoder = Decoder_Chairs(config)

        if args.model == "vae":
            model = VAE(config, encoder, decoder).cuda()
        elif args.model == "vamp":
            model = VAMP(config, encoder, decoder).cuda()
        elif args.model == "iwae":
            model = IWAE(config, encoder, decoder).cuda()

        ### Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate, eps=1e-4)

        ### Scheduler
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[50, 100, 125, 150],
            gamma=0.5,
            verbose=True
        )

    elif args.dataset == "starmen":

        encoder = Encoder_Chairs(config)
        decoder = Decoder_Chairs(config)

        if args.model == "vae":
            model = VAE(config, encoder, decoder).cuda()
        elif args.model == "vamp":
            model = VAMP(config, encoder, decoder).cuda()
        elif args.model == "iwae":
            model = IWAE(config, encoder, decoder).cuda()

        ### Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate, eps=1e-4)

        ### Scheduler
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[50, 100, 125, 150],
            gamma=0.5,
            verbose=True
        )

    elif args.dataset == "rotated_mnist":

        encoder = Encoder_ColorMNIST(config)
        decoder = Decoder_ColorMNIST(config)

        if args.model == "vae":
            model = VAE(config, encoder, decoder).cuda()
        elif args.model == "vamp":
            model = VAMP(config, encoder, decoder).cuda()
        elif args.model == "iwae":
            model = IWAE(config, encoder, decoder).cuda()

        ### Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate, eps=1e-4)

        ### Scheduler
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[50, 75, 100, 125, 150],
            gamma=10**(-1/4),
            verbose=True
        )


    logger.info("Successfully loaded data !\n")
    logger.info("------------------------------------------------------------")
    logger.info("Dataset \t \t Shape \t \t \t Range")
    logger.info(
        f"{args.dataset.upper()} train data: \t {train_data.shape} \t [{train_data.min()}-{train_data.max()}] "
    )
    logger.info(
        f"{args.dataset.upper()} eval data: \t {eval_data.shape} \t [{eval_data.min()}-{eval_data.max()}]"
    )
    logger.info("------------------------------------------------------------\n")

    logger.info(model)

    train_dataset = My_MaskedDataset(train_data, train_seq_mask, train_pix_mask)
    eval_dataset = My_MaskedDataset(eval_data, eval_seq_mask, eval_pix_mask)

    callbacks = []

    if args.use_wandb:
        from pythae.trainers.training_callbacks import WandbCallback

        wandb_cb = WandbCallback()
        wandb_cb.setup(
            training_config,
            model_config=config,
            project_name=args.wandb_project,
            entity_name=args.wandb_entity,
        )

        wandb_cb._wandb.config.update({"args": vars(args)})

        callbacks.append(wandb_cb)

    trainer = BaseTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_config=training_config,
            optimizer=optimizer,
            scheduler=scheduler,
            callbacks=callbacks,
        )

    trainer.train()

    if args.compute_nll:

        if args.dataset == "colormnist":
            test_data = torch.load(os.path.join('/home/clement/Documents/TopographicVAE/tvae/data/test_12_long_color_mnist.pt'), map_location="cpu")

        elif args.dataset == "3d_chairs":
            test_data = torch.load(os.path.join('/gpfswork/rech/wlr/uhw48em/rvae/data/3d_chairs/3D_chairs_color.pt'), map_location="cpu")[1200:]

        elif args.dataset == "sprites":
            test_data = torch.load(os.path.join('/gpfswork/rech/wlr/uhw48em/rvae/data/sprites/Sprites_test.pt'), map_location="cpu")['data']

        elif args.dataset == "starmen":
            test_data = torch.load(os.path.join('/gpfswork/rech/wlr/uhw48em/rvae/data/starmen/starmen_1000.pt'), map_location="cpu")[900:].float()

        elif args.dataset == "rotated_mnist":
            test_data = torch.load(os.path.join('/gpfswork/rech/wlr/uhw48em/rvae/data/rotated_mnist/test_mnist_rotated_8.pt'), map_location="cpu")


        ### Reload model
        trained_model = AutoModel.load_from_folder(os.path.join(training_config.output_dir, f'{trainer.model.model_name}_training_{trainer._training_signature}', 'final_model')).to(device).eval()

        from evaluation import evaluate_model_likelihood
        test_nll = evaluate_model_likelihood(model=trained_model, test_data=test_data.to(device), n_samples=args.nll_n_samples, batch_size=100)

        if args.use_wandb:
            from pythae.trainers.training_callbacks import WandbCallback

            wandb_cb = WandbCallback()
            wandb_cb.setup(
                training_config,
                model_config=config,
                project_name=args.wandb_project + "_metrics",
                entity_name=args.wandb_entity,
            )

            wandb_cb._wandb.config.update({"args": vars(args)})

            wandb_cb._wandb.log({
                "test/mean_nll": np.mean(test_nll),
                "test/std_nll": np.std(test_nll)
            })

            wandb_cb.on_train_end(training_config)


if __name__ == "__main__":
    print(args)
    main(args)
