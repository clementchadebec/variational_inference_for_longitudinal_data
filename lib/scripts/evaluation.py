import torch
import numpy as np
from tqdm.auto import tqdm
from utils import My_MaskedDataset
import torch.nn.functional as F



def _set_inputs_to_device(inputs):
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    inputs_on_device = inputs

    if device == "cuda":
        cuda_inputs = dict.fromkeys(inputs)

        for key in inputs.keys():
            if torch.is_tensor(inputs[key]):
                cuda_inputs[key] = inputs[key].cuda()

            else:
                cuda_inputs = inputs[key]
        inputs_on_device = cuda_inputs

    return inputs_on_device


def evaluate_model_reconstruction(model, test_dataset, batch_size, n_runs=5):
    
    test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

    test_recon = []
    for _ in range(n_runs):
        run_recon_loss = []
        with torch.no_grad():
            for inputs in tqdm(test_loader):
                inputs = _set_inputs_to_device(inputs)

                model_output = model(
                                inputs, epoch=np.inf, dataset_size=len(test_loader.dataset)
                            )

                recon_x = model_output.recon_x

                rec_loss = F.mse_loss(
                                recon_x.reshape(inputs['data'].shape[0]*inputs['data'].shape[1], -1),
                                inputs['data'].reshape(inputs['data'].shape[0]*inputs['data'].shape[1], -1),
                                reduction="none"
                            ).sum(dim=-1).mean(dim=0)

                run_recon_loss.append(rec_loss.item())
        test_recon.append(np.mean(run_recon_loss))

    return test_recon


def evaluate_model_reconstruction_of_missing(model, test_dataset, batch_size, n_runs=5):
    test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

    test_recon = []
    for _ in range(n_runs):
        run_recon_loss = []
        with torch.no_grad():
            for inputs in tqdm(test_loader):
                inputs = _set_inputs_to_device(inputs)

                model_output = model(
                                inputs, epoch=np.inf, dataset_size=len(test_loader.dataset)
                            )

                recon_x = model_output.recon_x
                
                # mse of missing pixels in seen images 
                rec_loss_pix = (
                    F.mse_loss(
                        recon_x.reshape(inputs['data'].shape[0] * inputs['data'].shape[1], -1),
                        inputs['data'].reshape(inputs['data'].shape[0] * inputs['data'].shape[1], -1),
                        reduction="none"
                    ) * (1-inputs['pix_mask']).reshape(inputs['data'].shape[0] * inputs['data'].shape[1], -1)
                ).sum(dim=-1)

                rec_loss_pix = (rec_loss_pix.reshape(inputs['data'].shape[0], -1) * inputs['seq_mask']).sum(dim=-1)

                # mse of missing images in sequences
                rec_loss_seq = (
                    F.mse_loss(
                        recon_x.reshape(inputs['data'].shape[0] * inputs['data'].shape[1], -1),
                        inputs['data'].reshape(inputs['data'].shape[0] * inputs['data'].shape[1], -1),
                        reduction="none"
                    ).reshape(inputs['data'].shape[0] * inputs['data'].shape[1], -1)
                ).sum(dim=-1)

                rec_loss_seq = (rec_loss_seq.reshape(inputs['data'].shape[0], -1) * (1 - inputs['seq_mask'])).sum(dim=-1)

                total_miss = (1 - inputs['pix_mask'] * inputs['seq_mask'].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).sum()

                run_recon_loss.append(rec_loss_seq.sum(dim=0).item() + rec_loss_pix.sum(dim=0).item())# / total_miss.item()) 
        test_recon.append(np.mean(run_recon_loss))
            

    return test_recon


def evaluate_model_reconstruction_of_missing_with_best_on_seen(model, test_dataset, batch_size, n_runs=5):
    test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

    test_recon = []
    for _ in range(n_runs):
        run_recon_loss = []
        with torch.no_grad():
            for inputs in tqdm(test_loader):
                inputs = _set_inputs_to_device(inputs)

                rec, idx = model.infer_missing(
                    x=inputs['data'],
                    seq_mask=inputs["seq_mask"],
                    pix_mask=inputs["pix_mask"]

                )

                recon_x = torch.cat([rec[idx[0, i], i].unsqueeze(0) for i in range(rec.shape[1])])
                
                # mse of missing pixels in seen images 
                rec_loss_pix = (
                    F.mse_loss(
                        recon_x.reshape(inputs['data'].shape[0] * inputs['data'].shape[1], -1),
                        inputs['data'].reshape(inputs['data'].shape[0] * inputs['data'].shape[1], -1),
                        reduction="none"
                    ) * (1-inputs['pix_mask']).reshape(inputs['data'].shape[0] * inputs['data'].shape[1], -1)
                ).sum(dim=-1)

                rec_loss_pix = (rec_loss_pix.reshape(inputs['data'].shape[0], -1) * inputs['seq_mask']).sum(dim=-1)

                # mse of missing images in sequences
                rec_loss_seq = (
                    F.mse_loss(
                        recon_x.reshape(inputs['data'].shape[0] * inputs['data'].shape[1], -1),
                        inputs['data'].reshape(inputs['data'].shape[0] * inputs['data'].shape[1], -1),
                        reduction="none"
                    ).reshape(inputs['data'].shape[0] * inputs['data'].shape[1], -1)
                ).sum(dim=-1)

                rec_loss_seq = (rec_loss_seq.reshape(inputs['data'].shape[0], -1) * (1 - inputs['seq_mask'])).sum(dim=-1)

                total_miss = (1 - inputs['pix_mask'] * inputs['seq_mask'].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).sum()

                run_recon_loss.append(rec_loss_seq.sum(dim=0).item() + rec_loss_pix.sum(dim=0).item())# / total_miss.item()) 
        test_recon.append(np.mean(run_recon_loss))
            

    return test_recon

def evaluate_model_likelihood(model, test_data, n_samples=1, batch_size=100, n_runs=5):
    test_nll = []
    for _ in range(n_runs):
        if model.model_name == "VAE" or model.model_name == "VAMP" or model.model_name == "IWAE":
            test_nll.append(model.get_nll(data=test_data.reshape((-1,)+model.model_config.input_dim), n_samples=n_samples, batch_size=batch_size))
        else:
            test_nll.append(model.get_nll(data=test_data, n_samples=n_samples, batch_size=batch_size))
    return test_nll