from typing import Union
import torch
from tqdm import tqdm
from libs.config import Config


def _infer_aleatoric(model, dataloader, device):
    model.eval()  # no dropout when inferring aleatoric uncertaity

    total_result_dict = {}

    mu_result, sigma_result = [], []
    for _ in tqdm(range(1), desc="Doing inference 1 time for aleatoric..."):
        for batch in dataloader:
            x = batch["data"]
            x = x.to(device)
            mu, sigma = model(x)
            prob = torch.softmax(mu, dim=1)  # prob.shape == (N, 2)

            mu_result.append(prob.detach().cpu())
            sigma_result.append(sigma.detach().cpu())

    total_result_dict["mu"] = torch.cat(mu_result, dim=0).detach().cpu().numpy()
    total_result_dict["sigma"] = torch.cat(sigma_result, dim=0).detach().cpu().numpy()
    return total_result_dict


def _infer_epistemic(model, dataloader, device, num_mcd):
    model.train()  # make sure turning on dropout

    total_result = []
    for _ in tqdm(range(num_mcd), desc=f"Doing inference {num_mcd} times for epistemic..."):
        curr_result = []
        for batch in dataloader:
            x = batch["data"]
            x = x.to(device)
            mu, _ = model(x)  # mu.shape == (N, 2)
            prob = torch.softmax(mu, dim=1)  # prob.shape == (N, 2)
            prob_for_positive_class = prob[:, 1]
            curr_result.append(prob_for_positive_class.detach().cpu())
        total_result.append(torch.cat(curr_result, dim=0))

    result = torch.stack(total_result, dim=0)
    result = result.detach().cpu().numpy()
    return result


def infer(
    model,
    dataloader,
    device,
    num_mcd,
    config: Union[Config],
):
    include_aleatoric = config.uq_mode == "combined" or config.uq_mode == "aleatoric"
    include_epistemic = config.uq_mode == "combined" or config.uq_mode == "epistemic"

    aleatoric_result = None
    if include_aleatoric:
        aleatoric_result = _infer_aleatoric(model, dataloader, device)

    epistemic_result = None
    if include_epistemic:
        epistemic_result = _infer_epistemic(model, dataloader, device, num_mcd)

    return (aleatoric_result, epistemic_result)
