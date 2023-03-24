#!/bin/bash

python lvae_iaf.py --dataset colormnist --posterior gaussian --prior vamp --n_hidden_in_made 3 --n_made_blocks 2 --warmup 10 --vamp_number_components 150 --linear_scheduling_steps 0 --num_epochs 20 --learning_rate 1e-3 --batch_size 128 --steps_saving 25 --steps_saving 3 --use_wandb
