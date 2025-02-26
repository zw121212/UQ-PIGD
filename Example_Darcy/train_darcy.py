import pandas as pd
import torch
import torch.optim as optim
import os
from src.Diffusion import *
from src.UNet2D  import *
from src.TransCNN import *
from src.grad import *
from data_darcy import *
import logging
from tqdm import tqdm

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
device = torch.device("cuda:0" if torch.cuda.is_available()else 'cpu')

epsilon=1.e-10
batch=16
lr = 1.5e-4
epochs = 40000
num_steps=50
input_dim = 2
output_dim = 2
pixels_per_dim = 64

k_value = pd.read_csv('darcy/K_data.csv',nrows=3000).values.astype('float32')
p_value = pd.read_csv('darcy/p_data.csv',nrows=3000).values.astype('float32')
k_value = torch.tensor(k_value, dtype=torch.float32).to(device)
p_value = torch.tensor(p_value, dtype=torch.float32).to(device)

p_value += torch.randn_like(p_value)*torch.mean(p_value,dim=1,keepdim=True)*0.2
p = generalized_b_xy_c_to_image(p_value)
k = generalized_b_xy_c_to_image(k_value)
f=create_f(pixels_per_dim)
f= generalized_image_to_b_xy_c(f.unsqueeze(0)).to(device)# [1, pixels_per_dim*pixels_per_dim, 1]


dataset = PairedDataset(k, p)
dataloader = DataLoader(dataset,batch_size=batch,shuffle=True,num_workers=0)
model_Unet=UNet(dim=16,channels=1,sigmoid_last_channel=False).to(device)
model_trans=ResidualCNNNetwork(size=16).to(device)
diffusion=DenoisingDiffusion(n_steps=num_steps,device=device)
par = list(model_trans.parameters())+list(model_Unet.parameters())
opt = torch.optim.Adam(params=par, lr=lr)
scheduler = optim.lr_scheduler.StepLR(opt, step_size=15000, gamma=0.2)

mse = nn.MSELoss(reduction='none').to(device)
dict1=diffusion.diff_dict
pbar = tqdm(total=epochs // 10, desc="Epochs")
for epoch in range(epochs):
    k_0, p_0 = next(iter(dataloader))
    t = diffusion.sample_timesteps(k_0.shape[0]).to(device)
    k_t1, k_T = diffusion.q_sample(k_0, t + 1)
    alpha1 = (dict1['alpha'][t + 1]).unsqueeze(1).unsqueeze(1)
    alpha_hat = (dict1['alpha_hat'][t]).unsqueeze(1).unsqueeze(1)
    alpha_hat1 = (dict1['alpha_hat'][t + 1]).unsqueeze(1).unsqueeze(1)
    var = (dict1['var'][t + 1]).unsqueeze(1).unsqueeze(1)
    p_T = model_trans.forward(k_T)
    p_t1 = torch.sqrt(alpha_hat1) * p_0 + torch.sqrt(1 - alpha_hat1) * p_T
    p_t1.requires_grad_(True)
    a=p_t1 + k_0
    pre_p0 = model_Unet(a.unsqueeze(1), t + 1).squeeze()
    loss_vdm1 = mse(pre_p0, p_0).mean()

    ##微分算子
    pre_f ,residual_bc=Nx_Grad_2D(pre_p0, k_0, batch, output_dim, input_dim, pixels_per_dim,device)
    residual_in=pre_f-f
    residual=torch.cat([residual_in.unsqueeze(-1), residual_bc], dim=-1)  # [64,4096,3]

    loss_f=mse(residual, torch.zeros_like(residual)).mean(dim=[1,2])
    # loss_pde=(1 / (2 * var) * loss_f).mean()
    coeff = torch.sqrt(alpha_hat) * (1 - alpha1) / (1 - alpha_hat1)

    loss_pde1 = 1 / (2 * var.squeeze()) * coeff.squeeze() * loss_f
    pre_u_t = torch.sqrt(alpha_hat) * pre_p0 + torch.sqrt(1 - alpha_hat)* p_T
    J2 = torch.abs((1 - alpha_hat * torch.diff(pre_p0, dim=-1) / (torch.diff(pre_u_t, dim=-1) + epsilon)).mean(dim=-1, keepdim=True)
                   + (1 - alpha_hat * torch.diff(pre_p0, dim=1) / (torch.diff(pre_u_t, dim=1) + epsilon)).mean(dim=1, keepdim=True))/2

    loss_pde2 = torch.log(1 / (J2 + epsilon)).mean(dim=[1,2])
    loss_pde = (torch.max(loss_pde1 + loss_pde2, torch.zeros_like(loss_pde1) + epsilon)).mean()
    loss = loss_vdm1 * 10000 + loss_pde ** 0.3
    # loss=loss_vdm1*10000 + loss_pde**0.5
    opt.zero_grad()
    loss.backward()
    opt.step()
    scheduler.step()
    if epoch % 10 == 0:
        logging.info("\nCompleted epoch {}:".format(epoch + 1))
        pbar.set_postfix(
            loss=f"{loss.item()}",
            loss_vdm_u=f"{loss_vdm1.item()}",
            loss_pde=f"{loss_pde.item()}",
        )
        pbar.update(1)  # 更新进度条

    if (epoch + 1) % 2000 == 0:
        save_dir = "model"
        os.makedirs(save_dir, exist_ok=True)
        models = [model_Unet, model_trans, diffusion]
        models_para = {
            "model_Unet": model_Unet.state_dict(),
            "model_trans": model_trans.state_dict(),
        }
        torch.save(models_para, "{}/PIGD_Darcy_{}.pth".format(save_dir, epoch + 1))




