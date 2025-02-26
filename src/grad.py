import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from einops import rearrange
from findiff import FinDiff
def generalized_image_to_b_xy_c(tensor):
    """
    Transpose the tensor from [batch, channels, ..., pixel_x, pixel_y] to [batch, pixel_x*pixel_y, channels, ...]. We assume two pixel dimensions.
    """
    num_dims = len(tensor.shape) - 3  # subtracting batch and pixel dimensions
    pattern = 'b ' + ' '.join([f'c{i}' for i in range(num_dims)]) + ' x y -> b (x y) ' + ' '.join([f'c{i}' for i in range(num_dims)])
    return rearrange(tensor, pattern)
def generalized_b_xy_c_to_image(tensor, pixels_x=None, pixels_y=None):
    """
    Transpose the tensor from [batch, pixel_x*pixel_y, channels, ...] to [batch, channels, ..., pixel_x, pixel_y] using einops.
    """
    if pixels_x is None or pixels_y is None:
        pixels_x = pixels_y = int(np.sqrt(tensor.shape[1]))
    num_dims = len(tensor.shape) - 2  # Subtracting batch and pixel dimensions (NOTE that we assume two pixel dimensions that are FLATTENED into one dimension)
    pattern = 'b (x y) ' + ' '.join([f'c{i}' for i in range(num_dims)]) + f' -> b ' + ' '.join([f'c{i}' for i in range(num_dims)]) + ' x y'
    return rearrange(tensor, pattern, x=pixels_x, y=pixels_y)

class StencilGradientComputation(nn.Module):
    '''
    Warning: This is hard-coded for finite differences on images with 2nd order accuracy.
    '''

    def __init__(self, stencils, periodic=False, device='cpu'):
        super(StencilGradientComputation, self).__init__()

        # identify max kernel size
        self.max_inner_offset = 0
        self.max_offset = 0
        for key, stencil in stencils.items():
            for (i, j), value in stencil.items():
                if key == ('C', 'C'):
                    self.max_inner_offset = max(self.max_inner_offset, abs(i), abs(j))
                else:
                    self.max_offset = max(self.max_offset, abs(i), abs(j))
        self.max_inner_kernel_size = 2 * self.max_inner_offset + 1  # include center and in both directions
        self.max_kernel_size = 2 * self.max_offset + 1  # include center and in both directions

        self.kernels = {}
        mid_inner = self.max_inner_offset  # center of the kernel
        mid = self.max_offset  # center of the kernel
        for key, stencil in stencils.items():
            if key == ('C', 'C'):
                kernel = torch.zeros((1, 1, self.max_inner_kernel_size, self.max_inner_kernel_size), device=device)
                self.kernels[key] = kernel
                for (i, j), value in stencil.items():
                    kernel[0, 0, mid_inner + i, mid_inner + j] = value
            else:
                kernel = torch.zeros((1, 1, self.max_kernel_size, self.max_kernel_size), device=device)
                self.kernels[key] = kernel
                for (i, j), value in stencil.items():
                    kernel[0, 0, mid + i, mid + j] = value
            self.kernels[key] = kernel

            self.periodic = periodic

    def forward(self, x):

        original_size = x.size()
        batch_size, *channels, height, width = original_size

        # flatten the channel dimensions
        x = x.view(batch_size, -1, height, width)
        channels = x.size(1)

        interior_kernel = self.kernels[('C', 'C')]
        interior_kernel = interior_kernel.repeat((channels, 1, 1, 1))

        if self.periodic:
            # pad the image with the opposite boundary
            padding = (self.max_inner_offset, self.max_inner_offset, self.max_inner_offset, self.max_inner_offset)
            x = F.pad(x, padding, mode='circular')
            x_grads = F.conv2d(x, interior_kernel, groups=channels)
            return x_grads.view(original_size)

        interior_conv = F.conv2d(x, interior_kernel, groups=channels)

        # manually apply boundary stencils
        # we extend the image by max_offset since kernel is centered
        x_ext = F.pad(x, (self.max_offset, self.max_offset, self.max_offset, self.max_offset), mode='constant', value=0)

        # only consider the part of x that is at the boundary for the convolution (while being consistent with the convolution kernels)
        reduced_conv_offset = 2 * self.max_offset + self.max_inner_offset

        # top boundary
        top_kernel = self.kernels[('L', 'C')]
        top_kernel = top_kernel.repeat((channels, 1, 1, 1))
        top_conv = F.conv2d(x_ext[:, :, 0:reduced_conv_offset, :], top_kernel, groups=channels)

        # bottom boundary
        bottom_kernel = self.kernels[('H', 'C')]
        bottom_kernel = bottom_kernel.repeat((channels, 1, 1, 1))
        bottom_conv = F.conv2d(x_ext[:, :, -reduced_conv_offset:, :], bottom_kernel, groups=channels)

        # left boundary
        left_kernel = self.kernels[('C', 'L')]
        left_kernel = left_kernel.repeat((channels, 1, 1, 1))
        left_conv = F.conv2d(x_ext[:, :, :, 0:reduced_conv_offset], left_kernel, groups=channels)

        # right boundary
        right_kernel = self.kernels[('C', 'H')]
        right_kernel = right_kernel.repeat((channels, 1, 1, 1))
        right_conv = F.conv2d(x_ext[:, :, :, -reduced_conv_offset:], right_kernel, groups=channels)

        # top-left corner
        tl_corner_kernel = self.kernels[('L', 'L')]
        tl_corner_kernel = tl_corner_kernel.repeat((channels, 1, 1, 1))
        tl_corner_conv = F.conv2d(x_ext[:, :, 0:reduced_conv_offset, 0:reduced_conv_offset], tl_corner_kernel,
                                  groups=channels)

        # top-right corner
        tr_corner_kernel = self.kernels[('L', 'H')]
        tr_corner_kernel = tr_corner_kernel.repeat((channels, 1, 1, 1))
        tr_corner_conv = F.conv2d(x_ext[:, :, 0:reduced_conv_offset, -reduced_conv_offset:], tr_corner_kernel,
                                  groups=channels)

        # bottom-left corner
        bl_corner_kernel = self.kernels[('H', 'L')]
        bl_corner_kernel = bl_corner_kernel.repeat((channels, 1, 1, 1))
        bl_corner_conv = F.conv2d(x_ext[:, :, -reduced_conv_offset:, 0:reduced_conv_offset], bl_corner_kernel,
                                  groups=channels)

        # bottom-right corner
        br_corner_kernel = self.kernels[('H', 'H')]
        br_corner_kernel = br_corner_kernel.repeat((channels, 1, 1, 1))
        br_corner_conv = F.conv2d(x_ext[:, :, -reduced_conv_offset:, -reduced_conv_offset:], br_corner_kernel,
                                  groups=channels)

        # combine the results from interior, boundaries, and corners
        x_grads = torch.zeros_like(x)
        x_grads[:, :, self.max_inner_offset:-self.max_inner_offset,
        self.max_inner_offset:-self.max_inner_offset] = interior_conv
        x_grads[:, :, 0:self.max_inner_offset, :] = top_conv
        x_grads[:, :, -self.max_inner_offset:, :] = bottom_conv
        x_grads[:, :, :, 0:self.max_inner_offset] = left_conv
        x_grads[:, :, :, -self.max_inner_offset:] = right_conv
        x_grads[:, :, 0:self.max_inner_offset, 0:self.max_inner_offset] = tl_corner_conv
        x_grads[:, :, 0:self.max_inner_offset, -self.max_inner_offset:] = tr_corner_conv
        x_grads[:, :, -self.max_inner_offset:, 0:self.max_inner_offset] = bl_corner_conv
        x_grads[:, :, -self.max_inner_offset:, -self.max_inner_offset:] = br_corner_conv

        # reshape back to the original dimensions
        x_grads = x_grads.view(original_size)
        return x_grads


class StencilGradients(nn.Module):
    '''
    This is hard-coded for finite differences on images with n-th order accuracy (for first and second derivatives).
    '''

    def __init__(self, d0=1, d1=1, fd_acc=2, periodic=False, device='cpu'):
        super(StencilGradients, self).__init__()
        self.d_d0 = StencilGradientComputation(FinDiff(0, d0, 1, acc=fd_acc).stencil((99, 99)).data, periodic, device)
        self.d_d0 = StencilGradientComputation(FinDiff(0, d0, 1, acc=fd_acc).stencil((99, 99)).data, periodic, device)
        self.d_d1 = StencilGradientComputation(FinDiff(1, d1, 1, acc=fd_acc).stencil((99, 99)).data, periodic, device)
        self.d_d00 = StencilGradientComputation(FinDiff(0, d0, 2, acc=fd_acc).stencil((99, 99)).data, periodic, device)
        self.d_d11 = StencilGradientComputation(FinDiff(1, d1, 2, acc=fd_acc).stencil((99, 99)).data, periodic, device)
        self.d_d01 = StencilGradientComputation(FinDiff((0, d0, 1), (1, d1, 1), acc=fd_acc).stencil((99, 99)).data,
                                                periodic, device)

    def forward(self, x, mode):
        if mode == 'all':
            return self.d_d0(x), self.d_d1(x), self.d_d00(x), self.d_d11(x), self.d_d01(x)
        elif mode == 'd_d0':
            return self.d_d0(x)
        elif mode == 'd_d1':
            return self.d_d1(x)
        elif mode == 'd_d00':
            return self.d_d00(x)
        elif mode == 'd_d11':
            return self.d_d11(x)
        elif mode == 'd_d01':
            return self.d_d01(x)
        else:
            raise NotImplementedError
def Nx_Grad_2D(p_0,k_0,batch, output_dim, input_dim, pixels_per_dim,device):
    domain_length = 1;
    pixels_per_dim = 64;
    periodic = False
    d0 = domain_length / (pixels_per_dim - 1)
    d1 = domain_length / (pixels_per_dim - 1)
    fd_acc = 2
    stencil_gradients = StencilGradients(d0=d0, d1=d1, fd_acc=fd_acc, periodic=periodic, device=device)
    p=p_0
    permeability_field = k_0  # HACK remove gradients here?!
    p_d0 = stencil_gradients(p, mode='d_d0')
    p_d1 = stencil_gradients(p, mode='d_d1')
    grad_p = torch.stack([p_d0, p_d1], dim=-3)
    p_d00 = stencil_gradients(p, mode='d_d00')
    p_d11 = stencil_gradients(p, mode='d_d11')
    perm_d0 = stencil_gradients(permeability_field, mode='d_d0')
    perm_d1 = stencil_gradients(permeability_field, mode='d_d1')
    grad_p = generalized_image_to_b_xy_c(grad_p)
    velocity_jacobian = torch.zeros(batch, output_dim, input_dim, pixels_per_dim, pixels_per_dim, device=device,
                                    dtype=torch.float32)
    velocity_jacobian[:, 0, 0] = -permeability_field * p_d00 - perm_d0 * p_d0
    velocity_jacobian[:, 1, 1] = -permeability_field * p_d11 - perm_d1 * p_d1
    velocity_jacobian = generalized_image_to_b_xy_c(velocity_jacobian)
    # obtain equilibrium equations for residual
    eq_0 = velocity_jacobian[:, :, 0, 0] + velocity_jacobian[:, :, 1, 1]

    # manually add BCs   添加边界条件：根据计算的梯度信息，为边界条件手动设置残差
    # reshape output to match image shape
    grad_p_img = generalized_b_xy_c_to_image(grad_p)
    residual_bc = torch.zeros_like(grad_p_img)
    residual_bc[:, 0, 0, :] = -grad_p_img[:, 0, 0, :]  # xmin / top (acc. to matplotlib visualization)
    residual_bc[:, 0, -1, :] = grad_p_img[:, 0, -1, :]  # xmax / bot

    residual_bc[:, 1, :, 0] = grad_p_img[:, 1, :, 0]  # ymin / left
    residual_bc[:, 1, :, -1] = -grad_p_img[:, 1, :, -1]  # ymax / right

    residual_bc = generalized_image_to_b_xy_c(residual_bc)
    # residual = torch.cat([eq_0.unsqueeze(-1), residual_bc], dim=-1)

    return eq_0,residual_bc
class StencilGradientComputation1D(nn.Module):
    def __init__(self, stencils,periodic=False,device='cpu'):
        super(StencilGradientComputation1D, self).__init__()

        self.device = device
        self.periodic = periodic
        # Calculate maximum kernel size
        self.max_offset = 0
        for key, stencil in stencils.items():
            for (i,) in stencil.keys():
                self.max_offset = max(self.max_offset, abs(i))
        self.max_kernel_size = 2 * self.max_offset + 1

        self.kernels = {}
        mid = self.max_offset  # Center of the kernel
        for key, stencil in stencils.items():
            kernel = torch.zeros((1, 1, self.max_kernel_size), device=device)
            for (i,), value in stencil.items():
                kernel[0, 0, mid + i] = value
            self.kernels[key] = kernel

    def forward(self, x):
        original_size = x.size()
        batch_size, channels, length = original_size

        # Flatten the channel dimension
        x = x.view(batch_size, -1, length)
        channels = x.size(1)

        # Get gradient kernels
        interior_kernel = self.kernels[('C',)]
        interior_kernel = interior_kernel.repeat((channels, 1, 1))

        if self.periodic:
            # Periodic boundary conditions: wrap around
            padding = self.max_offset
            x_padded = F.pad(x, (padding, padding), mode='circular')
        else:
            # Zero padding for boundaries
            padding = self.max_offset
            x_padded = F.pad(x, (padding, padding), mode='constant', value=0)

        # Convolution to compute gradients
        x_grads = F.conv1d(x, interior_kernel, groups=channels, padding=self.max_offset)

        return x_grads.view(original_size)

class StencilGradients1D(nn.Module):
    def __init__(self, d_0=1,fd_acc=2, periodic=False, device='cpu'):
        super(StencilGradients1D, self).__init__()
        self.d_1=StencilGradientComputation1D(FinDiff(0,d_0, 1, acc=fd_acc).stencil((99,)).data, periodic=periodic, device=device)
        self.d_2=StencilGradientComputation1D(FinDiff(0,d_0, 2, acc=fd_acc).stencil((99,)).data, periodic=periodic, device=device)

    def forward(self,x,mode):
        if mode=='d_1':
            return self.d_1(x)
        if mode=='d_2':
            return self.d_2(x)
        else:
            raise NotImplementedError


if __name__ == '__main__':
    device = 'cpu'
    d0 = 64;
    d1 = 64;
    fd_acc = 2;
    periodic = False
    stencil_gradients = StencilGradients(d0=d0, d1=d1, fd_acc=fd_acc, periodic=periodic, device=device)
