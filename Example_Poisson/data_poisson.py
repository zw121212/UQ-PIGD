"""
This file is used for generating solution data for the Poisson equation: -Δu=f，𝑓~𝐺𝑃(1/2,9/400*exp(−25(𝑥−𝑥′)^2))
The source term f is first sampled from a Gaussian field,
and then the equation is discretized using the finite difference method to generate the solution data.
"""
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd


class GP_F :
    def __init__(self, num_x_samples):
        self.observations = {"x": list(), "y": list()}
        self.num_x_samples = num_x_samples
        self.x_samples = np.arange(-1.0, 1.0, 2.0 / self.num_x_samples).reshape(-1, 1)

        # prior
        self.mu = np.zeros_like(self.x_samples)
        self.mu += 0.5
        self.cov = self.kernel(self.x_samples, self.x_samples)

    def visualize(self, num_gp_samples=1000):
        gp_samples = np.random.multivariate_normal(
            mean=self.mu.ravel(),
            cov=self.cov,
            size=num_gp_samples)
        x_sample = self.x_samples.ravel()

        # plt.figure()
        value = np.zeros((num_gp_samples, self.num_x_samples))
        for i, gp_sample in enumerate(gp_samples):
            value[i, :] = gp_sample
            # plt.plot(x_sample, gp_sample)
        # for x_pos in x_sample:
        #     plt.axvline(x=x_pos, color='gray', linestyle='--')
        # plt.title('sample paths of f(x)')
        #plt.legend()
        #plt.grid()
        ### 'Plot the sampled images'
        plt.show()
        return value

    @staticmethod
    def kernel(x1, x2, l=1 / np.sqrt(50), sigma_f=3 / 20):
        dist_matrix = np.sum(x1 ** 2, 1).reshape(-1, 1) + np.sum(x2 ** 2, 1) - 2 * np.dot(x1, x2.T)  # 两点的距离
        return sigma_f ** 2 * np.exp(-0.5 / l ** 2 * dist_matrix)

num_x_samples=64
gp = GP_F(num_x_samples=num_x_samples)
f_value = torch.tensor(gp.visualize(),dtype=torch.float32)
"""
 The sample matrix has a shape of num_gp_samples * num_x_samples,  
 where num_gp_samples represents the number of samples, and num_x_samples represents the positions of the samples.

"""


'use the finite difference method to generate the solution data'
coefficients_matrix=torch.zeros((f_value.shape[1],f_value.shape[1]))
coefficients_matrix[0,0]=coefficients_matrix[-1,-1]=1

for i in range(1,f_value.shape[1]-1):
    coefficients_matrix[i,i]=2
    coefficients_matrix[i,i-1]=coefficients_matrix[i,i+1]=-1
coefficients_matrix=coefficients_matrix/(2/f_value.shape[1])**2
u=torch.linalg.solve(coefficients_matrix, f_value,left=False)
u[:,0]=u[:,-1]=0

# plt.figure()
# for j in range(f_value.shape[0]):
#     plt.plot(np.linspace(-1, 1, f_value.shape[1]), u[j,:])
#     plt.axvline(x=-1.0, color='gray', linestyle='--')
#     plt.axvline(x=1.0, color='gray', linestyle='--')
#     plt.title('sample paths of u(x)')
# plt.show()


# df=pd.DataFrame(f_value)
# df1=pd.DataFrame(u)
# df1.to_csv('data_Poisson/u_poisson_64.csv',index=False,header=False)
# df.to_csv('data_Poisson/f_poisson_64.csv',index=False,header=False)




