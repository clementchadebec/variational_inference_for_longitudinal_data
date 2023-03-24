import os
from posixpath import basename

import torch
import logging
from pythae.models import VAEConfig, VAE, LVAE_IAF, LVAE_IAF_Config, AutoModel
from pythae.trainers import BaseTrainerConfig, BaseTrainer
import argparse
import numpy as np

from utils import Encoder_HMNIST, Decoder_HMNIST, My_MaskedDataset, Encoder_Sprites_Missing, Decoder_Sprites_Missing

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)

ap = argparse.ArgumentParser()

# Training setting
ap.add_argument(
    "--dataset",
    type=str,
    default="hmnist",
    choices=["hmnist", "sprites"],
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
        "lvae_iaf"
    ],
    default="lvae_iaf"
)
ap.add_argument(
    "--prior",
    choices=[
        "standard",
        "vamp"
    ],
    default="standard"
)
ap.add_argument(
    "--posterior",
    choices=[
        "gaussian",
        "iaf"
    ],
    default="gaussian"
)
ap.add_argument(
    "--n_hidden_in_made",
    type=int,
    default=3
)
ap.add_argument(
    "--n_made_blocks",
    type=int,
    default=2
)
ap.add_argument(
    "--warmup",
    type=int,
    default=10
)
ap.add_argument(
    "--gradient_clip",
    type=float,
    default=1e4
)
ap.add_argument(
    "--context_dim",
    type=int,
    default=None
)
ap.add_argument(
    "--vamp_number_components",
    type=int,
    default=500
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
    "--compute_mse",
    action="store_true",
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

    if args.dataset == "hmnist":
        args.input_dim = (1, 28, 28)
        args.reconstruction_loss = "bce"
        data = np.load("/gpfswork/rech/wlr/uhw48em/rvae/data/rotated_mnist/missing_for_gpvae_60.npz")
        

        x_train_full = torch.tensor(np.moveaxis(data['x_train_full'], 4, 2))
        x_train_miss = torch.tensor(np.moveaxis(data['x_train_miss'], 4, 2))
        m_train_miss = torch.tensor(np.moveaxis(data['m_train_miss'], 4, 2)).type(torch.bool)

        x_val_full = torch.tensor(np.moveaxis(data['x_test_full'], 4, 2))
        x_val_miss = torch.tensor(np.moveaxis(data['x_test_miss'], 4, 2))
        m_val_miss = torch.tensor(np.moveaxis(data['m_test_miss'], 4, 2)).type(torch.bool)
      
    elif args.dataset == "sprites":
        args.input_dim = (3, 64, 64)
        args.reconstruction_loss = "mse"
        data = np.load('/home/clement/Documents/GP-VAE/data/sprites/sprites.npz')

        x_train_full = np.moveaxis(data['x_train_full'].reshape(-1, 8, 64, 64, 3), 4, 2)
        x_train_miss = np.moveaxis(data['x_train_miss'].reshape(-1, 8, 64, 64, 3), 4, 2)
        m_train_miss = np.moveaxis(data['m_train_miss'].reshape(-1, 8, 64, 64, 3), 4, 2)

        x_val_full = torch.tensor(x_train_full[8000:]).float()
        x_val_miss = torch.tensor(x_train_miss[8000:]).float()
        m_val_miss = torch.tensor(m_train_miss[8000:]).type(torch.bool)

        x_train_full = torch.tensor(x_train_full[:8000]).float()
        x_train_miss = torch.tensor(x_train_miss[:8000]).float()
        m_train_miss = torch.tensor(m_train_miss[:8000]).type(torch.bool)


    train_seq_mask = torch.ones(x_train_miss.shape[:2], requires_grad=False).type(torch.bool)
    eval_seq_mask = torch.ones(x_val_miss.shape[:2], requires_grad=False).type(torch.bool)

    if args.model == "lvae_iaf":
        config = LVAE_IAF_Config(
            input_dim=args.input_dim,
            reconstruction_loss=args.reconstruction_loss,
            n_obs_per_ind=x_train_miss.shape[1],
            latent_dim=args.latent_dim,
            n_hidden_in_made=args.n_hidden_in_made,
            n_made_blocks=args.n_made_blocks,
            warmup=args.warmup,
            context_dim=args.context_dim,
            prior=args.prior,
            posterior=args.posterior,
            vamp_number_components=args.vamp_number_components,
            linear_scheduling_steps=args.linear_scheduling_steps
        )

    else:
        config = VAEConfig(
            input_dim=args.input_dim,
            latent_dim=args.latent_dim,
            reconstruction_loss=args.reconstruction_loss,
        )

    training_config = BaseTrainerConfig(
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        steps_saving=args.steps_saving,
        steps_predict=args.steps_predict,
        shuffle=shuffle_data,
        gradient_clip=args.gradient_clip,
        keep_best_on_train=True
    )

    if args.dataset == "hmnist":

        encoder = Encoder_HMNIST(config)
        decoder = Decoder_HMNIST(config)

    elif args.dataset == "sprites":
        encoder = Encoder_Sprites_Missing(config)
        decoder = Decoder_Sprites_Missing(config)

    if args.model == "lvae_iaf":
        model = LVAE_IAF(config, encoder, decoder).cuda()
    else:
        model = VAE(config, encoder, decoder).cuda()

    ### Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate, eps=1e-4)

    ### Scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[200],
        gamma=10**(-1/4),
        verbose=True
    )


    logger.info("Successfully loaded data !\n")
    logger.info("------------------------------------------------------------")
    logger.info("Dataset \t \t Shape \t \t \t Range")
    logger.info(
        f"{args.dataset.upper()} train data: \t {x_train_miss.shape} \t [{x_train_miss.min()}-{x_train_miss.max()}] "
    )
    logger.info(
        f"{args.dataset.upper()} eval data: \t {x_val_miss.shape} \t [{x_val_miss.min()}-{x_val_miss.max()}]"
    )
    logger.info("------------------------------------------------------------\n")

    logger.info(model)
    logger.info("------------------------------------------------------------")
    logger.info(f"Encoder num params: {sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)}")
    logger.info(f"Decoder num params: {sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)}")
    logger.info("------------------------------------------------------------")
    train_dataset = My_MaskedDataset(x_train_full, train_seq_mask, ~m_train_miss)
    eval_dataset = My_MaskedDataset(x_val_full, eval_seq_mask, ~m_val_miss)

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

    if args.compute_mse:

        if args.dataset == "hmnist":
            test_data = np.load('/gpfswork/rech/wlr/uhw48em/rvae/data/rotated_mnist/missing_for_gpvae_60.npz')
        

            x_test_full = torch.tensor(np.moveaxis(data['x_test_full'], 4, 2))
            x_test_miss = torch.tensor(np.moveaxis(data['x_test_miss'], 4, 2))
            m_test_miss = torch.tensor(np.moveaxis(data['m_test_miss'], 4, 2)).type(torch.bool)

        elif args.dataset == "sprites":
            data = np.load('/home/clement/Documents/GP-VAE/data/sprites/sprites.npz')

            x_test_full = torch.tensor(np.moveaxis(data['x_test_full'].reshape(-1, 8, 64, 64, 3), 4, 2)).float()
            x_test_miss = torch.tensor(np.moveaxis(data['x_test_miss'].reshape(-1, 8, 64, 64, 3), 4, 2)).float()
            m_test_miss = torch.tensor(np.moveaxis(data['m_test_miss'].reshape(-1, 8, 64, 64, 3), 4, 2)).type(torch.bool)


        test_seq_mask = torch.ones(x_test_miss.shape[:2], requires_grad=False).type(torch.bool)

        ### Reload model
        trained_model = AutoModel.load_from_folder(os.path.join(training_config.output_dir, f'{trainer.model.model_name}_training_{trainer._training_signature}', 'final_model')).to(device).eval()

        from evaluation import evaluate_model_reconstruction_of_missing
        test_mse = evaluate_model_reconstruction_of_missing(model=trained_model, test_data=x_test_full.to(device), test_seq_mask=test_seq_mask, test_pix_mask=~m_test_miss, batch_size=100, binary=False)[0]
        print((m_test_miss.sum(), test_mse.sum()))
        mean_mse = torch.sum(test_mse) / m_test_miss.sum()
        std_mse = (1 / (m_test_miss.sum() - 1) * ((test_mse - mean_mse)**2).sum()).sqrt()
        print("Mean mse: ", mean_mse)
        print("std mse: ", std_mse)

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
                "test/mean_mse": mean_mse
            })

            wandb_cb.on_train_end(training_config)

if __name__ == "__main__":
    print(args)
    main(args)
