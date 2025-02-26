import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.Unet1D import *
from src.Diffusion import *
from src.TransMLP import  *
from matplotlib import rcParams
from matplotlib.patches import Rectangle

device = torch.device("cuda:0" if torch.cuda.is_available()else 'cpu')
rcParams['axes.unicode_minus'] = False
plt.rcParams["font.family"] = "Times New Roman"

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

f_value = pd.read_csv('f_variable_128.csv',skiprows=300).values.astype('float32')
u_value = pd.read_csv('u_variable_128.csv',skiprows=300).values.astype('float32')
f_value = torch.tensor(f_value, dtype=torch.float32).to(device)
u_value = torch.tensor(u_value, dtype=torch.float32).to(device)
# l=torch.randint(low=0,high=300, size=(1,)).to(device)
# l=l.item()
l=5  # random select index
f_0=f_value[l,:]
u_0=u_value[l,:]

batch = 16
n_steps = 50
model_Unet = Unet1d(batch)
model_trans = TransMLP_fu(128, 128,5)
diffusion = DenoisingDiffusion(n_steps=n_steps, device='cpu')

checkpoint = torch.load("PIGD_2_{}.pth".format(30000), map_location=device)
model_Unet.load_state_dict(checkpoint['model_Unet'])
model_trans.load_state_dict(checkpoint['model_trans'])
model_Unet = model_Unet.to('cpu')
model_trans = model_trans.to('cpu')
diffusion = diffusion.to('cpu')
mse = torch.nn.MSELoss()
model_trans.eval()
diffusion.eval()
dict1 = diffusion.diff_dict

N = 10
f_0 = f_0.unsqueeze(0).unsqueeze(0)
u_0 = u_0.unsqueeze(0).unsqueeze(0)
with (torch.no_grad()):
    pre_u0_list = []
    for k in range(N):
        f_T = torch.randn(1, 1, 128)
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
            u_t1=torch.sqrt(alpha_hat) * pre_u0 + torch.sqrt(1 - alpha_hat) * u_T+std*torch.randn_like(pre_u0)
        pre_u0_list.append(pre_u0.detach().cpu())
pre_u0_tensor = torch.stack(pre_u0_list)
mean_pre_u0 = (pre_u0_tensor.mean(dim=0)).squeeze()
var_pre_u0 = (pre_u0_tensor.var(dim=0)).squeeze()
std_pre_u0 = torch.sqrt(var_pre_u0)
print('std_mean:', std_pre_u0.mean().item())
absolute_error = torch.abs(mean_pre_u0.squeeze() - u_0.squeeze())
relative_error = torch.sqrt(absolute_error**2/ torch.sqrt((u_0**2).sum())) * 100
mse_error = torch.mean(absolute_error**2)
print('L inf error: ', absolute_error.max().item())
print("abs error:", (absolute_error.mean()).item())
print("relative error (%):", (relative_error.mean()).item())
print('L2 error',torch.sqrt(mse_error).item())


zoom_x =torch.linspace(0,0.1,6)
zoom_y = mean_pre_u0[64:70]
region = (torch.min(zoom_x),torch.max(zoom_x),torch.min(zoom_y),torch.max(zoom_y))  # (x_min, x_max, y_min, y_max)

# 创建主图
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(np.linspace(-1,1,128), mean_pre_u0, label='Mean', color='blue')
ax.fill_between(np.linspace(-1,1,128), mean_pre_u0 - 1.96*std_pre_u0, mean_pre_u0 + 1.96*std_pre_u0,
                 color='lightblue', alpha=0.5, label='Variance Interval')
ax.plot(np.linspace(-1,1,128), u_0[0,0,:], label='True', color='red')
ax.set_xlabel(r"$x$",fontfamily="Times New Roman")
ax.set_ylabel(r"$u(x)$",fontfamily="Times New Roman")
ax.legend()
# 添加虚线框标注放大区域
rect = Rectangle((region[0], region[2]), region[1] - region[0], region[3] - region[2],
                 linewidth=1.5, edgecolor="black", linestyle="--", facecolor="none")
ax.add_patch(rect)

# 确定箭头的起点和终点
arrow_x = (region[0] + region[1]) / 2  # 放大区域的横向中心
arrow_y = (region[2] + region[3]) / 2  # 放大区域的纵向中心
sub_ax_center = (0.5, -2)  # 子图中心

# 添加箭头
ax.annotate(
    "", xy=(arrow_x, arrow_y), xytext=sub_ax_center,
    arrowprops=dict(facecolor="black", arrowstyle="->", lw=1.5)
)

# 添加放大的子图
sub_ax = fig.add_axes([0.64, 0.2, 0.25, 0.25])  # 子图位置：左,下,宽,高 (相对主图比例)
sub_ax.plot(np.linspace(0,0.1,6), mean_pre_u0[64:70], label='Mean', color='blue')
sub_ax.fill_between(np.linspace(0,0.1,6), mean_pre_u0[64:70] - 1.96*std_pre_u0[64:70], mean_pre_u0[64:70] + 1.96*std_pre_u0[64:70],
                 color='lightblue', alpha=0.5, label='Variance Interval')
sub_ax.plot(np.linspace(0,0.1,6), u_0[0,0,64:70], label='True', color='red')
sub_ax.set_xlabel(r"$x$",fontfamily="Times New Roman")
sub_ax.set_ylabel(r"$u(x)$",fontfamily="Times New Roman")
sub_ax.tick_params(axis='both', which='major', labelsize=8)
plt.show()