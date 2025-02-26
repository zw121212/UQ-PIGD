import torch
import pandas as pd
from src.UNet2D import *
from src.grad import *
from src.Diffusion import *
from src.TransCNN import *
from matplotlib import rcParams
import matplotlib.pyplot as plt
from data_darcy import *
device = torch.device("cuda:0" if torch.cuda.is_available()else 'cpu')

rcParams['axes.unicode_minus'] = False
df = pd.read_csv('darcy/K_data_test.csv')
data= df.iloc[:].values
k_value=torch.tensor(data,dtype=torch.float32)
df1 = pd.read_csv('darcy/p_data_test.csv')
data1 = df1.iloc[:].values
p_value=torch.tensor(data1,dtype=torch.float32)
print(p_value.shape)
device = torch.device("cuda:0" if torch.cuda.is_available()else 'cpu')

num_steps=50
input_dim = 2
output_dim = 2#输出有两个通道分别表示的是K,P
pixels_per_dim = 64
p = generalized_b_xy_c_to_image(p_value)
k = generalized_b_xy_c_to_image(k_value)
f=create_f(pixels_per_dim)
f= generalized_image_to_b_xy_c(f.unsqueeze(0)).to(device)# [1, pixels_per_dim*pixels_per_dim, 1]
model_Unet=UNet(dim=16,channels=1,sigmoid_last_channel=False).to(device)
model_trans=ResidualCNNNetwork(size=16).to(device)
diffusion=DenoisingDiffusion(n_steps=num_steps,device=device)
checkpoint = torch.load("PIGD_Darcy_{}.pth".format(40000), map_location=device)
model_Unet.load_state_dict(checkpoint['model_Unet'])
model_trans.load_state_dict(checkpoint['model_trans'])
model_Unet = model_Unet.to('cpu')
model_trans = model_trans.to('cpu')
diffusion = diffusion.to('cpu')
mse = torch.nn.MSELoss()
model_trans.eval()
diffusion.eval()
dict1 = diffusion.diff_dict

n_steps=50
num_samples=64
# l = torch.randint(low=0, high=999, size=(1,))
# l = l.item()
l = 3
plt.figure()
k0 = k_value[l, :].reshape(num_samples, num_samples)
x1_samples = np.linspace(0, 1.0, num_samples)
x2_samples = np.linspace(0, 1.0, num_samples)
X1, X2 = np.meshgrid(x1_samples, x2_samples)
x_samples = np.vstack([X1.ravel(), X2.ravel()]).T
plt.imshow(k0, extent=[x_samples.min(), x_samples.max(), x_samples.min(), x_samples.max()],
           origin='lower', cmap='magma', alpha=0.8)
plt.colorbar(label="Value")
plt.title("Permeability ")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
z0 = p_value[l, :].reshape(num_samples, num_samples)
plt.figure()
plt.imshow(z0, extent=[x_samples.min(), x_samples.max(), x_samples.min(), x_samples.max()],
           origin='lower', cmap='magma', alpha=0.8)
plt.colorbar(label="Value")
plt.title("Pressure p --True")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

N = 5
with (torch.no_grad()):
    pre_u0_list = []
    for n in range(N):
        k_T = torch.randn(1,64,64)
        p_T = model_trans.forward(k_T)
        k_t1 = k_T
        p_t1 = p_T
        pre_p0 = model_Unet((p_T + k[l,:,:]).unsqueeze(1), torch.tensor([49])).squeeze()
        plt.figure()
        z_sample = pre_p0.reshape(num_samples, num_samples)

        for i in range(1, n_steps-1):
            p_t1.requires_grad_(True)
            t = torch.tensor([n_steps - 1 - i])
            std =torch.sqrt(dict1['var'][t])
            alpha1 = (dict1['alpha'][t + 1]).unsqueeze(1).unsqueeze(1)
            alpha_hat = (dict1['alpha_hat'][t]).unsqueeze(1).unsqueeze(1)
            alpha_hat1 = (dict1['alpha_hat'][t + 1]).unsqueeze(1).unsqueeze(1)
            pre_p0 = model_Unet((p_t1 + k[l,:,:]).unsqueeze(1), torch.tensor([49])).squeeze()
            u_t1=torch.sqrt(alpha_hat) * pre_p0 + torch.sqrt(1 - alpha_hat) * p_T+std*torch.randn_like(pre_p0)
        pre_u0_list.append(z_sample.detach().cpu())
    pre_u0_tensor = torch.stack(pre_u0_list)
    mean_pre_u0 = (pre_u0_tensor.mean(dim=0)).squeeze()
    var_pre_u0 = (pre_u0_tensor.var(dim=0)).squeeze()
    std_pre_u0 = torch.sqrt(var_pre_u0)
    print('std_mean', std_pre_u0.mean().item())
    plt.imshow(mean_pre_u0, extent=[x_samples.min(), x_samples.max(), x_samples.min(), x_samples.max()],
               origin='lower', cmap='magma', alpha=0.8)
    plt.colorbar(label="Value")
    plt.title("Pressure mean  --Pre")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    error=torch.abs(mean_pre_u0-z0)
    plt.figure()
    plt.imshow(error, extent=[x_samples.min(), x_samples.max(), x_samples.min(), x_samples.max()],
               origin='lower', cmap='magma', alpha=0.8)
    plt.colorbar(label="Value")
    plt.title("abs_error")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    plt.imshow(std_pre_u0, extent=[x_samples.min(), x_samples.max(), x_samples.min(), x_samples.max()],
               origin='lower', cmap='magma', alpha=0.8)
    plt.colorbar(label="Value")
    plt.title("std")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    relative_error = torch.sqrt(error ** 2 / torch.sqrt((z0 ** 2).sum())) * 100
    mse_error = torch.mean(error ** 2)
    print('L inf error: ', error.max().item())
    print(f'L2 error: {torch.sqrt(mse_error):.8f}')
    print(f'L1 error: {error.mean():,.8f}')
    print(f'Relative error: {relative_error.mean():.8f}')



