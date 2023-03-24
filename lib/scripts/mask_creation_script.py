import torch
import numpy as np
import os
from utils import make_batched_masks
import argparse
import os

ap = argparse.ArgumentParser()

# Training setting
ap.add_argument(
    "--dataset",
    type=str,
    default="colormnist",
    choices=["colormnist", "3d_chairs", "sprites", "starmen", "rotated_mnist", "faces", "faces_64"],
    help="Path to dataset.",
    required=True,
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
    "--batch_size",
    type=int,
    default=128
)


args = ap.parse_args()

def main(args):


    if args.dataset == "sprites":
        input_dim = (3, 64, 64)
        train_data = torch.load(os.path.join('/gpfswork/rech/wlr/uhw48em/rvae/data//sprites/Sprites_train.pt'), map_location="cpu")['data'][:-1000]
        eval_data = torch.load(os.path.join('/gpfswork/rech/wlr/uhw48em/rvae/data//sprites/Sprites_train.pt'), map_location="cpu")['data'][-1000:]
        test_data = torch.load(os.path.join('/gpfswork/rech/wlr/uhw48em/rvae/data//sprites/Sprites_test.pt'), map_location="cpu")['data']

    elif args.dataset == "starmen":
        input_dim = (1, 64, 64)
        train_data = torch.load(os.path.join('/gpfswork/rech/wlr/uhw48em/rvae/data//starmen/starmen_1000.pt'), map_location="cpu")[:700]
        eval_data = torch.load(os.path.join('/gpfswork/rech/wlr/uhw48em/rvae/data//starmen/starmen_1000.pt'), map_location="cpu")[700:900]
        test_data = torch.load(os.path.join('/gpfswork/rech/wlr/uhw48em/rvae/data//starmen/starmen_1000.pt'), map_location="cpu")[900:].float()

    elif args.dataset == "rotated_mnist":
        input_dim = (1, 28, 28)
        train_data = torch.load(os.path.join('/gpfswork/rech/wlr/uhw48em/rvae/data//rotated_mnist/train_mnist_rotated_8.pt'), map_location="cpu")[:-1000]
        eval_data = torch.load(os.path.join('/gpfswork/rech/wlr/uhw48em/rvae/data//rotated_mnist/train_mnist_rotated_8.pt'), map_location="cpu")[-1000:]
        test_data = torch.load(os.path.join('/gpfswork/rech/wlr/uhw48em/rvae/data//rotated_mnist/test_mnist_rotated_8.pt'), map_location="cpu")


    elif args.dataset == "colormnist":
        input_dim = (1, 28, 28)
        train_data = torch.load(os.path.join('/gpfswork/rech/wlr/uhw48em/rvae/data/colormnist/train_12_long_color_mnist.pt'), map_location="cpu")#[:10000]
        eval_data = torch.load(os.path.join('/gpfswork/rech/wlr/uhw48em/rvae/data/colormnist/val_12_long_color_mnist.pt'), map_location="cpu")#[:5000]
        test_data = torch.load(os.path.join('/gpfswork/rech/wlr/uhw48em/rvae/data/colormnist/test_12_long_color_mnist.pt'), map_location="cpu")



    if args.prob_missing_data > 0.:

        train_seq_mask = make_batched_masks(train_data, args.prob_missing_data, args.batch_size).type(torch.bool)
        eval_seq_mask = make_batched_masks(eval_data, args.prob_missing_data, args.batch_size).type(torch.bool)
        test_seq_mask = make_batched_masks(test_data, args.prob_missing_data, args.batch_size).type(torch.bool)

        print(f'\nPercentage of missing data in train: {1 - train_seq_mask.sum() / np.prod(train_seq_mask.shape)} (target: {args.prob_missing_data})')
        print(f'Percentage of missing data in eval: {1 - eval_seq_mask.sum() / np.prod(eval_seq_mask.shape)} (target: {args.prob_missing_data})')
        print(f'Percentage of missing data in test: {1 - test_seq_mask.sum() / np.prod(test_seq_mask.shape)} (target: {args.prob_missing_data})')
        shuffle_data = False

    else:
        train_seq_mask = torch.ones(train_data.shape[:2], requires_grad=False).type(torch.bool)
        eval_seq_mask = torch.ones(eval_data.shape[:2], requires_grad=False).type(torch.bool)
        test_seq_mask = torch.ones(test_data.shape[:2], requires_grad=False).type(torch.bool)

    if args.prob_missing_pixels > 0.:
        train_pix_mask = torch.distributions.Bernoulli(probs=1-args.prob_missing_pixels).sample((train_data.shape[0], train_data.shape[1],)+train_data.shape[-2:]).unsqueeze(2).repeat(1, 1, train_data.shape[2], 1, 1)
        eval_pix_mask = torch.distributions.Bernoulli(probs=1-args.prob_missing_pixels).sample((eval_data.shape[0], eval_data.shape[1],)+eval_data.shape[-2:]).unsqueeze(2).repeat(1, 1, eval_data.shape[2], 1, 1)
        test_pix_mask = torch.distributions.Bernoulli(probs=1-args.prob_missing_pixels).sample((test_data.shape[0], test_data.shape[1],)+test_data.shape[-2:]).unsqueeze(2).repeat(1, 1, test_data.shape[2], 1, 1)
        print(f'\nPercentage of missing pixels in train: {1 - train_pix_mask.sum() / np.prod(train_pix_mask.shape)} (target: {args.prob_missing_pixels})')
        print(f'Percentage of missing pixels in eval: {1 - eval_pix_mask.sum() / np.prod(eval_pix_mask.shape)} (target: {args.prob_missing_pixels})')
        print(f'Percentage of missing pixels in test: {1 - test_pix_mask.sum() / np.prod(test_pix_mask.shape)} (target: {args.prob_missing_pixels})')

    else:
        train_pix_mask = torch.ones_like(train_data, requires_grad=False).type(torch.bool)
        eval_pix_mask = torch.ones_like(eval_data, requires_grad=False).type(torch.bool)
        test_pix_mask = torch.ones_like(test_data, requires_grad=False).type(torch.bool)


    data_path = "/gpfswork/rech/wlr/uhw48em/rvae/data/"

    np.savez(
        data_path + f'{args.dataset}/masks/mask_miss_data_{args.prob_missing_data}_miss_pixels_{args.prob_missing_pixels}.npz',
        train_seq_mask=train_seq_mask,
        eval_seq_mask=eval_seq_mask,
        test_seq_mask=test_seq_mask,
        train_pix_mask=train_pix_mask,
        eval_pix_mask=eval_pix_mask,
        test_pix_mask=test_pix_mask
    )

if __name__ == "__main__":
    print(args)
    main(args)
