import time
from src.Unet1D import *
from src.Diffusion import *
from src.TransMLP import *
from matplotlib import rcParams
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
device = torch.device("cuda:0" if torch.cuda.is_available()else 'cpu')
rcParams['axes.unicode_minus'] = False
plt.rcParams["font.family"] = "Times New Roman"
f_value = pd.read_csv('f_poisson_64.csv',skiprows=300).values.astype('float32')
u_value = pd.read_csv('u_poisson_64.csv',skiprows=300).values.astype('float32')
f_value = torch.tensor(f_value, dtype=torch.float32).to(device)
u_value = torch.tensor(u_value, dtype=torch.float32).to(device)
# l=torch.randint(low=0,high=300, size=(1,)).to(device)
# l=l.item()
l=5  # random select index
f_0=f_value[l,:].unsqueeze(0)
u_0=u_value[l,:].unsqueeze(0)


# ###设置网络
batch = 16
n_steps = 50
size = 64
model_Unet = Unet1d(batch)
diffusion = DenoisingDiffusion(n_steps=n_steps, device='cpu')
model_trans = TransMLP_fu(size, size, 5).to(device)

checkpoint = torch.load("PIGD_1_{}.pth".format(21000), map_location=device)
model_Unet.load_state_dict(checkpoint['model_Unet'])
model_trans.load_state_dict(checkpoint['model_trans'])
model_Unet = model_Unet.to('cpu')
model_trans = model_trans.to('cpu')
mse = torch.nn.MSELoss()
diffusion.eval()
model_trans.eval()
dict1 = diffusion.diff_dict

f_0 = f_0.unsqueeze(0)
u_0 = u_0.unsqueeze(0)
print(f_0.shape,u_0.shape)
N = 20
with (torch.no_grad()):
    pre_u0_list = []
    for k in range(N):
        if k%10==0:
            print(k)
        f_T = torch.randn(1, 1, 64)
        u_T = model_trans.forward(f_T)
        f_t1 = f_T
        u_t1 = u_T
        pre_u0 = model_Unet(u_T + f_0, torch.tensor([49]))
        for i in range(1, n_steps-1):
            t = torch.tensor([n_steps - 1 - i])
            std = torch.sqrt(dict1['var'][t])
            alpha1 = (dict1['alpha'][t + 1]).unsqueeze(1).unsqueeze(1)
            alpha_hat = (dict1['alpha_hat'][t]).unsqueeze(1).unsqueeze(1)
            alpha_hat1 = (dict1['alpha_hat'][t + 1]).unsqueeze(1).unsqueeze(1)
            pre_u0 = model_Unet(u_t1 + f_0, t + 1)
            u_t1=torch.sqrt(alpha_hat) * pre_u0 + torch.sqrt(1 - alpha_hat) * u_T + std*torch.randn_like(pre_u0)
        pre_u0_list.append(pre_u0.detach().cpu())
pre_u0_tensor = torch.stack(pre_u0_list)
mean_pre_u0 = (pre_u0_tensor.mean(dim=0)).squeeze()
var_pre_u0 = (pre_u0_tensor.var(dim=0)).squeeze()
std_pre_u0 = torch.sqrt(var_pre_u0)
print('std_mean', std_pre_u0.mean().item())

x = np.linspace(-1,1,64)
plt.figure()
plt.plot(np.linspace(-1,1,64), mean_pre_u0, label='Mean', color='blue')
plt.fill_between(np.linspace(-1,1,64), mean_pre_u0 - 1.96*std_pre_u0, mean_pre_u0 + 1.96*std_pre_u0,
                 color='lightblue', alpha=0.5, label='Variance Interval')
plt.plot(np.linspace(-1,1,64), u_0[0,0,:], label='True', color='red')
plt.legend()
plt.show()

absolute_error = torch.abs(mean_pre_u0 - u_0.squeeze())
relative_error = torch.sqrt(absolute_error**2/ torch.sqrt((u_0**2).sum())) * 100
mse_error = torch.sqrt(torch.mean(absolute_error**2))
print('L inf error: ', absolute_error.max().item())
print("abs error:", (absolute_error.mean()).item())
print("relative (%):", (relative_error.mean()).item())
print('L2 error', mse_error.item())
print('std-mean error: ', std_pre_u0.mean().item())

zoom_x = torch.linspace(-0.1,0.1,7)
zoom_y = mean_pre_u0[28:35]
region = (torch.min(zoom_x),torch.max(zoom_x),0.24,0.26)  # (x_min, x_max, y_min, y_max)
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(np.linspace(-1,1,64), mean_pre_u0, label='Mean', color='blue')
ax.fill_between(np.linspace(-1,1,64), mean_pre_u0 - 1.96*std_pre_u0, mean_pre_u0 + 1.96*std_pre_u0,
                 color='lightblue', alpha=0.5, label='Variance Interval')
ax.plot(np.linspace(-1,1,64), u_0[0,0,:], label='True', color='red')
ax.set_xlabel(r"$x$",fontfamily="Times New Roman")
ax.set_ylabel(r"$u(x)$",fontfamily="Times New Roman")
ax.legend()
rect = Rectangle((region[0], region[2]), region[1] - region[0], region[3] - region[2],
                 linewidth=1.5, edgecolor="black", linestyle="--", facecolor="none")
ax.add_patch(rect)
arrow_x = (region[0] + region[1]) / 2
arrow_y = (region[2] + region[3]) / 2
sub_ax_center = (0.3, 0.1)

ax.annotate(
    "", xy=(arrow_x, arrow_y), xytext=sub_ax_center,
    arrowprops=dict(facecolor="black", arrowstyle="->", lw=1.5)
)

sub_ax = fig.add_axes([0.5, 0.2, 0.25, 0.25])
sub_ax.plot(np.linspace(-0.1,0.1,7), mean_pre_u0[28:35], label='Mean', color='blue')
sub_ax.fill_between(np.linspace(-0.1,0.1,7), mean_pre_u0[28:35] - 1.96*std_pre_u0[28:35], mean_pre_u0[28:35] + 1.96*std_pre_u0[28:35],
                 color='lightblue', alpha=0.5, label='Variance Interval')
sub_ax.plot(np.linspace(-0.1,0.1,7), u_0[0,0,28:35], label='True', color='red')
sub_ax.set_xlabel(r"$x$",fontfamily="Times New Roman")
sub_ax.set_ylabel(r"$u(x)$",fontfamily="Times New Roman")
sub_ax.tick_params(axis='both', which='major', labelsize=8)
plt.show()