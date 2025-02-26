import torch
import pandas as pd
def u(x, omega):
    g = (x**2 - 1)
    h = sum(
        omega[2*i-2] * torch.sin(i * torch.pi * x) + omega[2*i-1] * torch.cos(i * torch.pi * x)
        for i in range(1, 5)
    )
    return g * h

def du_dx(x, omega):
    g = (x**2 - 1)
    g_prime = 2 * x
    h = sum(
        omega[2*i-2] * torch.sin(i * torch.pi * x) + omega[2*i-1] * torch.cos(i * torch.pi * x)
        for i in range(1, 5)
    )
    h_prime = sum(
        omega[2*i-2] * i * torch.pi * torch.cos(i * torch.pi * x) -
        omega[2*i-1] * i * torch.pi * torch.sin(i * torch.pi * x)
        for i in range(1, 5)
    )
    return g_prime * h + g * h_prime

def d2u_dx2(x, omega):
    g = (x**2 - 1)
    g_prime = 2 * x
    g_double_prime = 2
    h = sum(
        omega[2*i-2] * torch.sin(i * torch.pi * x) + omega[2*i-1] * torch.cos(i * torch.pi * x)
        for i in range(1, 5)
    )
    h_prime = sum(
        omega[2*i-2] * i * torch.pi * torch.cos(i * torch.pi * x) -
        omega[2*i-1] * i * torch.pi * torch.sin(i * torch.pi * x)
        for i in range(1, 5)
    )
    h_double_prime = sum(
        -omega[2*i-2] * (i * torch.pi)**2 * torch.sin(i * torch.pi * x) -
        omega[2*i-1] * (i * torch.pi)**2 * torch.cos(i * torch.pi * x)
        for i in range(1, 5)
    )
    return g_double_prime * h + 2 * g_prime * h_prime + g * h_double_prime

def v_prime(x, omega):
    u_prime = du_dx(x, omega)
    u_double_prime = d2u_dx2(x, omega)
    out=u_prime + (x + 2) * u_double_prime
    return -out/100

x=torch.linspace(-1,1,128)
u_values=[]
f_values=[]
for i in range(400):
    omega = torch.rand(8)
    u_value=u(x,omega)
    f_value=v_prime(x,omega)
    u_values.append(u_value)
    f_values.append(f_value)
u_values=torch.stack(u_values)
f_values=torch.stack(f_values)

df=pd.DataFrame(f_values)
df1=pd.DataFrame(u_values)
df1.to_csv('u_variable_128.csv',index=False,header=False)
df.to_csv('f_variable_128.csv',index=False,header=False)
