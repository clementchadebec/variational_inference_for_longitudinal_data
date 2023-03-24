import os

import torch
import logging
from pythae.models import GPVAE, GPVAEConfig, AutoModel
from pythae.trainers import BaseTrainerConfig, BaseTrainer
import argparse
import numpy as np

from utils import Encoder_Chairs_GPVAE, Encoder_Chairs_GPVAE, Decoder_ColorMNIST, Decoder_Chairs, Encoder_ColorMNIST_GPVAE, My_MaskedDataset, make_batched_masks

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
    "--prob_missing_data",
    type=float,
    default=0.,
    help='The probability of missing data (sequences will have at least 2 data)'
)
ap.add_argument(
    "--prob_missing_pixels",
    type=float,
    default=0.,
    help='The probability of missing pixels in the images'
)
ap.add_argument(
    "--kernel",
    type=str,
    choices=["cauchy", "rbf", "diffusion", "matern"],
    default="cauchy"
)
ap.add_argument(
    "--kernel_scales",
    type=int,
    default=1
)
ap.add_argument(
    "--sigma",
    type=float,
    default=1.
)
ap.add_argument(
    "--length_scale",
    type=float,
    default=1.
)
ap.add_argument(
    "--beta",
    type=float,
    default=1.
)
ap.add_argument(
    "--out_channels_time_cnn",
    type=int,
    default=265
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
    "--compute_mse",
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
    default="gpvae",
)
ap.add_argument(
    "--wandb_entity",
    help="wandb entity name",
    default="your-name",
)

args = ap.parse_args()


def main(args):

    device = "cuda" if torch.cuda.is_available() else 'cpu'
    shuffle_data = True 

    if args.dataset == "colormnist":
        args.input_dim = (3, 28, 28)
        train_data = torch.load(os.path.join('../../data/colormnist/train_12_long_color_mnist.pt'), map_location="cpu")#[:10000]
        eval_data = torch.load(os.path.join('../../data/colormnist/val_12_long_color_mnist.pt'), map_location="cpu")#[:5000]
        test_data = torch.load(os.path.join('../../data/colormnist/test_12_long_color_mnist.pt'), map_location="cpu")

    elif args.dataset == "3d_chairs":
        args.input_dim = (3, 64, 64)
        train_data = torch.load(os.path.join('../../data/3d_chairs/3D_chairs_color.pt'), map_location="cpu")[:1000]
        eval_data = torch.load(os.path.join('../../data/3d_chairs/3D_chairs_color.pt'), map_location="cpu")[1000:1200]
        test_data = torch.load(os.path.join('../../data/3d_chairs/3D_chairs_color.pt'), map_location="cpu")[1200:]

    elif args.dataset == "sprites":
        args.input_dim = (3, 64, 64)
        train_data = torch.load(os.path.join('../../data/sprites/Sprites_train.pt'), map_location="cpu")['data'][:-1000]
        eval_data = torch.load(os.path.join('../../data/sprites/Sprites_train.pt'), map_location="cpu")['data'][-1000:]
        test_data = torch.load(os.path.join('../../data/sprites/Sprites_test.pt'), map_location="cpu")['data']

    elif args.dataset == "starmen":
        args.input_dim = (1, 64, 64)
        train_data = torch.load(os.path.join('../../data/starmen/starmen_1000.pt'), map_location="cpu")[:700]
        eval_data = torch.load(os.path.join('../../data/starmen/starmen_1000.pt'), map_location="cpu")[700:900]
        test_data = torch.load(os.path.join('../../data/starmen/starmen_1000.pt'), map_location="cpu")[900:].float()

    elif args.dataset == "rotated_mnist":
        args.input_dim = (1, 28, 28)
        train_data = torch.load(os.path.join('../../data/rotated_mnist/train_mnist_rotated_8.pt'), map_location="cpu")[:-1000]
        eval_data = torch.load(os.path.join('../../data/rotated_mnist/train_mnist_rotated_8.pt'), map_location="cpu")[-1000:]
        test_data = torch.load(os.path.join('../../data/rotated_mnist/test_mnist_rotated_8.pt'), map_location="cpu")

    #if args.prob_missing_data > 0.:

    masks = np.load(f"../../data/{args.dataset}/masks/mask_miss_data_{args.prob_missing_data}_miss_pixels_{args.prob_missing_pixels}.npz")

    train_seq_mask=torch.from_numpy(masks["train_seq_mask"]).type(torch.bool)
    eval_seq_mask=torch.from_numpy(masks["eval_seq_mask"]).type(torch.bool)
    test_seq_mask=torch.from_numpy(masks["test_seq_mask"]).type(torch.bool)
    train_pix_mask=torch.from_numpy(masks["train_pix_mask"]).type(torch.bool)
    eval_pix_mask=torch.from_numpy(masks["eval_pix_mask"]).type(torch.bool)
    test_pix_mask=torch.from_numpy(masks["test_pix_mask"]).type(torch.bool)

    #train_seq_mask = make_batched_masks(train_data, args.prob_missing_data, args.batch_size).type(torch.bool)
    #eval_seq_mask = make_batched_masks(eval_data, args.prob_missing_data, args.batch_size).type(torch.bool)
    #test_seq_mask = make_batched_masks(test_data, args.prob_missing_data, args.batch_size).type(torch.bool)

    logger.info(f'\nPercentage of missing data in train: {1 - train_seq_mask.sum() / np.prod(train_seq_mask.shape)} (target: {args.prob_missing_data})')
    logger.info(f'Percentage of missing data in eval: {1 - eval_seq_mask.sum() / np.prod(eval_seq_mask.shape)} (target: {args.prob_missing_data})')
    logger.info(f'Percentage of missing data in test: {1 - test_seq_mask.sum() / np.prod(test_seq_mask.shape)} (target: {args.prob_missing_data})')
    
    logger.info(f'\nPercentage of missing pixels in train: {1 - train_pix_mask.sum() / np.prod(train_pix_mask.shape)} (target: {args.prob_missing_pixels})')
    logger.info(f'Percentage of missing pixels in eval: {1 - eval_pix_mask.sum() / np.prod(eval_pix_mask.shape)} (target: {args.prob_missing_pixels})')
    logger.info(f'Percentage of missing pixels in test: {1 - test_pix_mask.sum() / np.prod(test_pix_mask.shape)} (target: {args.prob_missing_pixels})')


    if args.prob_missing_data > 0.:
        shuffle_data = False


    config = GPVAEConfig(
        input_dim=args.input_dim,
        time_length=train_data.shape[1],
        kernel_choice=args.kernel,
        kernel_scales=args.kernel_scales,
        latent_dim=args.latent_dim,
        sigma=args.sigma,
        length_scale=args.length_scale,
        beta=args.beta,
        out_channels_time_cnn=args.out_channels_time_cnn
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

        encoder = Encoder_ColorMNIST_GPVAE(config)
        decoder = Decoder_ColorMNIST(config)

        model = GPVAE(config, encoder, decoder).cuda()

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

        encoder = Encoder_Chairs_GPVAE(config)
        decoder = Decoder_Chairs(config)

        model = GPVAE(config, encoder, decoder).cuda()

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

        encoder = Encoder_Chairs_GPVAE(config)
        decoder = Decoder_Chairs(config)

        model = GPVAE(config, encoder, decoder).cuda()

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

        encoder = Encoder_Chairs_GPVAE(config)
        decoder = Decoder_Chairs(config)

        model = GPVAE(config, encoder, decoder).cuda()

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

        encoder = Encoder_ColorMNIST_GPVAE(config)
        decoder = Decoder_ColorMNIST(config)

        model = GPVAE(config, encoder, decoder).cuda()

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
    logger.info(
        f"{args.dataset.upper()} test data: \t {test_data.shape} \t [{test_data.min()}-{test_data.max()}]"
    )
    logger.info("------------------------------------------------------------\n")

    logger.info(model)

    train_dataset = My_MaskedDataset(train_data, train_seq_mask, train_pix_mask)
    eval_dataset = My_MaskedDataset(eval_data, eval_seq_mask, eval_pix_mask)
    test_dataset = My_MaskedDataset(test_data, test_seq_mask, test_pix_mask)

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

    ### Reload model
    trained_model = AutoModel.load_from_folder(os.path.join(training_config.output_dir, f'{trainer.model.model_name}_training_{trainer._training_signature}', 'final_model')).to(device).eval()
    trained_model.device = device

    metrics = {}

    if args.compute_nll:

        from evaluation import evaluate_model_likelihood
        test_nll = evaluate_model_likelihood(model=trained_model, test_data=test_data.to(device), n_samples=args.nll_n_samples, batch_size=100)

        metrics['test/mean_nll'] = np.mean(test_nll)
        metrics['test/std_nll'] = np.std(test_nll)

    if args.compute_mse:

        from evaluation import evaluate_model_reconstruction, evaluate_model_reconstruction_of_missing
        test_recon = evaluate_model_reconstruction(model=trained_model, test_dataset=test_dataset, batch_size=args.batch_size)
        test_recon_missing = evaluate_model_reconstruction_of_missing(
            model=trained_model,
            test_dataset=test_dataset,
            batch_size=args.batch_size
            )

        metrics['test/mean_recon'] = np.mean(test_recon)
        metrics['test/std_recon'] = np.std(test_recon)

        metrics['test/mean_recon_missing'] = np.mean(test_recon_missing)
        metrics['test/std_recon_missing'] = np.std(test_recon_missing)

    print("Evaluation metric: ", metrics)

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

        wandb_cb._wandb.log(metrics)

        wandb_cb.on_train_end(training_config)

if __name__ == "__main__":
    print(args)
    main(args)
