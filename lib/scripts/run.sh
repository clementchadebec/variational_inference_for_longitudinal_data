#!/bin/bash

#python lvae_iaf.py --dataset colormnist --posterior gaussian --prior vamp --n_hidden_in_made 3 --n_made_blocks 2 --warmup 10 --vamp_number_components 150 --linear_scheduling_steps 0 --num_epochs 20 --learning_rate 1e-3 --batch_size 128 --steps_saving 25 --steps_saving 3 --use_wandb

python longvae.py --dataset colormnist --latent_dim 16 --prob_missing_data 0. --prob_missing_pixels 0. --num_epochs_vae 100 --num_epochs_longvae 100 --batch_size 128
python longvae.py --dataset sprites --latent_dim 16 --prob_missing_data 0. --prob_missing_pixels 0. --num_epochs_vae 100 --num_epochs_longvae 100 --batch_size 64
#python longvae.py --dataset sprites --latent_dim 16 --prob_missing_data 0.2 --prob_missing_pixels 0.3 --num_epochs_vae 1 --num_epochs_longvae 3


#python fid.py --dataset colormnist --model_path ../from_jz_lvae/models/colormnist/LVAE_IAF_training_2022-12-21_17-27-20/final_model/ --use_wandb
#python fid.py --dataset colormnist --model_path ../from_jz_lvae/models/colormnist/LVAE_IAF_training_2022-12-21_21-51-03/final_model/ --use_wandb
#python fid.py --dataset colormnist --model_path ../from_jz_lvae/models/colormnist/LVAE_IAF_training_2022-12-22_08-44-36/final_model/ --use_wandb
#python fid.py --dataset colormnist --model_path ../from_jz_lvae/models/colormnist/VAE_training_2022-12-14_17-00-38/final_model/ --use_wandb
#python fid.py --dataset colormnist --model_path ../from_jz_lvae/models/colormnist/VAMP_training_2022-12-14_16-59-53/final_model/ --use_wandb
#python fid.py --dataset colormnist --model_path ../from_jz_lvae/models/colormnist/GPVAE_training_2022-12-20_16-49-26/final_model/ --use_wandb
#python fid.py --dataset colormnist --model_path ../from_jz_lvae/models/colormnist/GPVAE_training_2022-12-20_17-44-15/final_model/ --use_wandb
#python fid.py --dataset colormnist --model_path ../from_jz_lvae/models/colormnist/GPVAE_training_2022-12-20_18-15-02/final_model/ --use_wandb
#python fid.py --dataset colormnist --model_path ../from_jz_lvae/models/colormnist/GPVAE_training_2022-12-20_18-39-44/final_model/ --use_wandb


#python fid.py --dataset sprites --model_path ../from_jz_lvae/models/sprites/LVAE_IAF_training_2022-12-13_18-46-18/final_model/ --use_wandb
#python fid.py --dataset sprites --model_path ../from_jz_lvae/models/sprites/LVAE_IAF_training_2022-12-13_19-26-27/final_model/ --use_wandb
#python fid.py --dataset sprites --model_path ../from_jz_lvae/models/sprites/LVAE_IAF_training_2022-12-21_22-05-50/final_model/ --use_wandb
#python fid.py --dataset sprites --model_path ../from_jz_lvae/models/sprites/VAE_training_2022-12-13_19-00-08/final_model/ --use_wandb
#python fid.py --dataset sprites --model_path ../from_jz_lvae/models/sprites/VAMP_training_2022-12-13_18-57-57/final_model/ --use_wandb
#python fid.py --dataset sprites --model_path ../from_jz_lvae/models/sprites/GPVAE_training_2022-12-20_17-12-22/final_model/ --use_wandb
#python fid.py --dataset sprites --model_path ../from_jz_lvae/models/sprites/GPVAE_training_2022-12-20_17-30-58/final_model/ --use_wandb
#python fid.py --dataset sprites --model_path ../from_jz_lvae/models/sprites/GPVAE_training_2022-12-20_18-10-57/final_model/ --use_wandb
#python fid.py --dataset sprites --model_path ../from_jz_lvae/models/sprites/GPVAE_training_2022-12-20_18-15-03/final_model/ --use_wandb


#python fid.py --dataset 3d_chairs --model_path ../from_jz_lvae/models/chairs/LVAE_IAF_training_2022-12-21_17-25-41/final_model/ --use_wandb
#python fid.py --dataset 3d_chairs --model_path ../from_jz_lvae/models/chairs/LVAE_IAF_training_2022-12-21_19-05-03/final_model/ --use_wandb
#python fid.py --dataset 3d_chairs --model_path ../from_jz_lvae/models/chairs/LVAE_IAF_training_2022-12-21_19-05-30/final_model/ --use_wandb
#python fid.py --dataset 3d_chairs --model_path ../from_jz_lvae/models/chairs/VAE_training_2022-12-13_18-59-17/final_model/ --use_wandb
#python fid.py --dataset 3d_chairs --model_path ../from_jz_lvae/models/chairs/VAMP_training_2022-12-13_18-55-39/final_model/ --use_wandb