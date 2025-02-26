import logging
import torch
import torch.nn as nn
from scipy.stats import norm
import torch.nn.init as init

from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class PairedDataset(Dataset):
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, idx):
        return self.data1[idx], self.data2[idx]


def set_models_to_eval(models):
    for model in models:
        model.eval()

def set_models_to_train(models):
    for model in models:
        model.train()


class DenoisingDiffusion(nn.Module):
    def __init__(self, n_steps, device):
        super(DenoisingDiffusion, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.diff_dict = self.create_diff_dict()
        self.coeff_sigma = torch.nn.Parameter(torch.zeros_like(self.diff_dict['alpha']), requires_grad=True)
        self.coeff_sigma = init.uniform_(self.coeff_sigma).to(self.device)
        self.coeff_mu = torch.nn.Parameter(self.diff_dict['alpha'].clone().detach().requires_grad_(True)).to(self.device)

    def make_beta_schedule(self, schedule='linear', n_timesteps=1000, start=1e-5, end=1e-2):
        if schedule == 'linear':
            betas = torch.linspace(start, end, n_timesteps)
        elif schedule == "quad":
            betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
        elif schedule == 'sigmoid':
            betas = torch.linspace(-6, 6, n_timesteps)
            betas = torch.sigmoid(betas) * (end - start) + start
        elif schedule == 'cosine':
            s = 0.008
            steps = n_timesteps + 1
            x = torch.linspace(0, n_timesteps, steps)
            alphas_cumprod = torch.cos(((x / n_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0, 0.999)
        return betas

    def create_diff_dict(self):
        diff_dict = {
            'betas': self.make_beta_schedule(schedule='cosine', n_timesteps=self.n_steps, start=1e-5, end=1e-2).to(
                self.device), }
        diff_dict['alpha'] = 1. - diff_dict['betas']
        diff_dict['alpha_hat'] = torch.cumprod(diff_dict['alpha'], 0)
        diff_dict['alpha_hat_1'] = torch.cat(
            [torch.tensor([1], device=self.device).float(), diff_dict['alpha_hat'][:-1]],
            0)
        diff_dict['var'] = (1 - diff_dict['alpha']) * (1 - diff_dict['alpha_hat_1']) / (1 - diff_dict['alpha_hat'])
        return diff_dict

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.n_steps-1, size=(n,))


    def q_sample(self, x_0, t, noisy=None):
        alpha_hat_t = self.diff_dict['alpha_hat'][t][:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.diff_dict['alpha_hat'][t])[:, None, None]
        if noisy == None:
            noisy1 =torch.randn_like(x_0).to(self.device)
        else:
            noisy1=noisy
        x_t = torch.sqrt(alpha_hat_t) * x_0 + sqrt_one_minus_alpha_hat * noisy1

        return x_t,noisy1
    def sample_val(self, models, model_u, n, size):
        logging.info(f"Sampling {n} new images....")
        set_models_to_eval(models)
        with torch.no_grad():
            noisy = torch.randn((n, 1,size)).to(self.device)
            noisy=models['model_trans'](noisy)
            for i in tqdm(reversed(range(1, self.n_steps)), position=0):
                t = torch.LongTensor([i] * noisy.shape[0]).to(self.device)

                predicted_noise = model_u(noisy, t).to(self.device)
                alpha = self.diff_dict['alpha'][t].view(-1, 1, 1)
                alpha_hat = self.diff_dict['alpha_hat'][t].view(-1,  1, 1)
                beta = self.diff_dict['betas'][t].view(-1, 1, 1)
                if i > 1:
                    noise = torch.randn(size=noisy.shape).to(self.device)
                else:
                    noise = torch.zeros_like(noisy).to(self.device)
                noisy = 1 / torch.sqrt(alpha) * (
                            noisy - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise  # u_theta
        set_models_to_train(models)
        return noisy


if __name__ == '__main__':
    nsteps = 10
    diffusion_try = DenoisingDiffusion(n_steps=nsteps, device='cpu')
    x_0 = torch.randn(5, 1, 256)
    t = diffusion_try.sample_timesteps(x_0.shape[0])
    x_t, noisy = diffusion_try.q_sample(x_0, t)
