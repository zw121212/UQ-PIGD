import torch
import os
from src.Diffusion import *
from src.TransMLP import *
from src.Unet1D import *
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
device = torch.device("cuda:0" if torch.cuda.is_available()else 'cpu')
def N_ax(u_value):
    u_value=u_value.squeeze()
    coefficients_matrix = torch.zeros((u_value.shape[-1], u_value.shape[-1]))
    coefficients_matrix[0, 0] = coefficients_matrix[-1, -1] = 1
    def a_x(x):
        return (x+2)/100
    h = 2 / u_value.shape[-1]

    for i in range(1, u_value.shape[-1] - 1):
        coefficients_matrix[i, i] = a_x(-1 + (i - 0.5) * h) + a_x(-1 + (i + 0.5) * h)
        coefficients_matrix[i, i - 1] = -a_x(-1 + (i - 0.5) * h)
        coefficients_matrix[i, i + 1] = -a_x(-1 + (i + 0.5) * h)

    coefficients_matrix = (coefficients_matrix / h ** 2).to(device)
    f = torch.matmul(coefficients_matrix, u_value.T).T
    return f.to(device)

f_value = pd.read_csv('f_variable_128.csv',nrows=300).values.astype('float32')
u_value = pd.read_csv('u_variable_128.csv',nrows=300).values.astype('float32')
f_value = torch.tensor(f_value, dtype=torch.float32).to(device)
u_value = torch.tensor(u_value, dtype=torch.float32).to(device)
u_value += torch.randn_like(u_value)*torch.mean(u_value,dim=1,keepdim=True)*0.2

epsilon=1.e-8
batch=16
size=128
lr = 1.5e-4
epochs = 30000
num_steps=50
dataset = PairedDataset(f_value, u_value)
dataloader = DataLoader(dataset,batch_size=batch,shuffle=True)

model_Unet = Unet1d(batch).to(device)
model_trans = TransMLP_fu(size, size, 5).to(device)
diffusion=DenoisingDiffusion(n_steps=num_steps,device=device)

par = list(model_trans.parameters())+list(model_Unet.parameters())
opt = torch.optim.Adam(params=par, lr=lr)
scheduler = optim.lr_scheduler.StepLR(opt, step_size=10000, gamma=0.2)

mse = nn.MSELoss(reduction='none').to(device)
dict1=diffusion.diff_dict
pbar = tqdm(total=epochs // 20, desc="Epochs")
for epoch in range(epochs):
    f_value, u_value = next(iter(dataloader))
    f_0 = f_value.unsqueeze(1).to(device)
    u_0=u_value.unsqueeze(1).to(device)
    t = diffusion.sample_timesteps(f_0.shape[0]).to(device)
    f_t1, f_T = diffusion.q_sample(f_0, t+1)
    alpha1 = (dict1['alpha'][t + 1]).unsqueeze(1).unsqueeze(1)
    alpha_hat =(dict1['alpha_hat'][t]).unsqueeze(1).unsqueeze(1)
    alpha_hat1 =(dict1['alpha_hat'][t+1]).unsqueeze(1).unsqueeze(1)
    var=dict1['var'][t+1].unsqueeze(1).unsqueeze(1)
    u_T = model_trans.forward(f_T)
    u_t1 = torch.sqrt(alpha_hat1) * u_0 + torch.sqrt(1 - alpha_hat1) * u_T
    u_t1.requires_grad_(True)

    pre_u0 = model_Unet(u_t1 + f_0, t + 1)
    pre_u_t = torch.sqrt(alpha_hat) * pre_u0 + torch.sqrt(1 - alpha_hat).unsqueeze(1).unsqueeze(1) * u_T
    pre_f0 = N_ax(pre_u0).unsqueeze(1)
    loss_vdm1 = mse(pre_u0, u_0).mean()
    loss_f =  mse(pre_f0[:,:,1:-1], f_0[:,:,1:-1]).mean(dim=[1, 2]).unsqueeze(1).unsqueeze(1)
    coeff = torch.sqrt(alpha_hat) * (1 - alpha1) / (1 - alpha_hat1)
    J2 = torch.abs(1 - alpha_hat * torch.diff(pre_u0) / (torch.diff(pre_u_t) + epsilon))

    loss_pde1 = 1 / (2 * var) * coeff * loss_f
    loss_pde2 = torch.log(1 / (J2 + epsilon))
    loss_pde = (torch.max(loss_pde1 + loss_pde2, torch.zeros_like(loss_pde1) + epsilon)).mean()
    loss = loss_vdm1 * 1000 + loss_pde**0.3

    opt.zero_grad()
    loss.backward()
    opt.step()
    scheduler.step()
    if epoch % 20 == 0:
        logging.info("\nCompleted epoch {}:".format(epoch + 1))
        pbar.set_postfix(
            loss=f"{loss.item():.6f}",
            loss_pde=f"{loss_pde.item():.6f}",
            loss_vdm_u=f"{loss_vdm1.item():.6f}",
        )
        pbar.update(1)  # 更新进度条
    if (epoch+1) % 3000== 0:
        save_dir = "model"
        os.makedirs(save_dir, exist_ok=True)
        models = [ model_Unet, model_trans, diffusion]
        models_para = {
            "model_Unet": model_Unet.state_dict(),
            "model_trans": model_trans.state_dict(),
            "diffusion": diffusion.state_dict(),
        }
        torch.save(models_para, "{}/PIGD_2_{}.pth".format(save_dir, epoch+1))


