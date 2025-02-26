import math
import torch
from torch import nn, einsum
from functools import partial
from einops import rearrange
from einops_exts import rearrange_many
from rotary_embedding_torch import RotaryEmbedding
import numpy as np


def generalized_b_xy_c_to_image(tensor, pixels_x=None, pixels_y=None):
    """
    Transpose the tensor from [batch, pixel_x*pixel_y, channels, ...] to [batch, channels, ..., pixel_x, pixel_y] using einops.
    """
    if pixels_x is None or pixels_y is None:
        pixels_x = pixels_y = int(np.sqrt(tensor.shape[1]))
    num_dims = len(tensor.shape) - 2  # Subtracting batch and pixel dimensions (NOTE that we assume two pixel dimensions that are FLATTENED into one dimension)
    pattern = 'b (x y) ' + ' '.join([f'c{i}' for i in range(num_dims)]) + f' -> b ' + ' '.join([f'c{i}' for i in range(num_dims)]) + ' x y'
    return rearrange(tensor, pattern, x=pixels_x, y=pixels_y)

def exists(x):
    return x is not None

def default(val, d):
    #用于处理默认参数值的情况，确保函数或程序在没有提供显式值时仍能正常工作。
    if exists(val):
        return val
    return d() if callable(d) else d

def is_odd(n):
    return (n % 2) == 1


def prob_mask_like(shape, prob, device):
    """
    生成一个给定形状和设备的布尔掩码张量，掩码中 True 的比例由 prob 参数控制
    """
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob



class Residual(nn.Module):
    #实现残差连接，通过将输入直接加到输出中来帮助训练深层网络
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class LayerNorm(nn.Module):
    '''实现对每个输入张量在维度 1 上进行归一化，只包含一个可训练的缩放参数 gamma。
    虽然与标准 LayerNorm 略有不同，但它提供了归一化和缩放的基本功能。'''
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps#防止分母为0
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


class PreNorm(nn.Module):
    '''PreNorm 是一个用于将归一化操作与功能模块结合的容器类。
    它先对输入张量进行归一化，然后将归一化后的张量传递给指定的功能模块 fn。
    这种设计模式有助于提高模型的训练稳定性'''
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)
class SinusoidalPosEmb(nn.Module):
    #是一个用于生成正弦位置编码的 PyTorch 模块
    # 将序列中的每个位置映射到一个具有唯一性的连续空间中，以便模型能够利用这些位置信息
    #正弦位置编码：提供了平滑且周期性的位置信息，能够处理不同长度的序列
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Block(nn.Module):
    #这个模块包括一个卷积层、一个归一化层和一个激活函数
    def __init__(self, dim, dim_out, padding_mode = 'zeros', groups = 8):
        super().__init__()
        if padding_mode == 'zeros' or padding_mode == 'circular':
            self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding = (0, 1, 1), padding_mode=padding_mode)
        else:
            raise ValueError('Unknown padding mode: {}'.format(padding_mode))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, padding_mode = 'zeros', groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists (time_emb_dim) else None

        self.block1 = Block(dim, dim_out, padding_mode = padding_mode, groups = groups)
        #会使用时间嵌入生成的缩放和平移参数
        self.block2 = Block(dim_out, dim_out, padding_mode = padding_mode, groups = groups)
        #对特征图进行进一步处理，通常包括卷积、归一化和激活函数操作
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
    def forward(self, x, time_emb = None):
        '''
        通过 h + self.res_conv(x) 实现残差连接，有助于缓解深层网络中的梯度消失问题，提高训练稳定性。
        残差连接可以使网络更容易学习恒等映射，进而提高模型的性能。
        '''
        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)

class SpatialLinearAttention(nn.Module):
    '''
    主要功能是通过注意力机制增强特征表示的能力
    使用多个注意力头来并行计算注意力，这样可以从不同的子空间中提取信息
    '''
    def __init__(self, dim, heads = 4, dim_head = 32, cond_dim = 64):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_q = nn.Conv2d(dim, hidden_dim, 1, bias = False)
        self.to_k = nn.Linear(cond_dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(cond_dim, hidden_dim, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = rearrange_many(qkv, 'b (h c) x y -> b h c (x y)', h = self.heads)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w) # added this (not included in original repo)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b = b)
class EinopsToAndFrom(nn.Module):
    '''
    主要用于在使用 einops 进行张量变换时，应用一个函数，并在应用前后对张量进行重新排列
    允许用户在张量的前向传播过程中灵活地改变张量的形状，以适应不同的操作需求。
    这种设计使得在处理复杂数据结构时能够方便地调整形状。

    '''
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x
class RelativePositionBias(nn.Module):
    def __init__(
        self,
        heads = 8,
        num_buckets = 32,
        max_distance = 128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets = 32, max_distance = 128):
        #将相对位置编码转换为固定大小的离散表示
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret
    """
    使用不同的方式处理小距离和大距离，
    使得较大的距离压缩到较少的bucket中。
    返回的张量 ret 包含相对位置对应的bucket索引
    """

    def forward(self, n, device):
        """
        计算查询和键之间的相对位置，然后根据相对位置计算相对注意力偏置
        """
        q_pos = torch.arange(n, dtype = torch.long, device = device)
        k_pos = torch.arange(n, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')#每对查询和键之间的相对位置
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets = self.num_buckets, max_distance = self.max_distance)#将相对位置映射到桶索引
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')
class Attention(nn.Module):
    '''实现了一个标准的多头自注意力机制，
    其中包括了旋转位置编码和条件输入的处理，以适应不同的任务需求'''
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        rotary_emb = None,
        cond_dim = 64,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)
        #将输入张量映射到查询（Q）、键（K）、值（V）的维度

        self.to_q = nn.Linear(dim, hidden_dim, bias = False)
        self.to_k = nn.Linear(cond_dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(cond_dim, hidden_dim, bias=False)
        #分别用于将输入张量、条件张量映射到查询、键、值的空间

        self.to_out = nn.Conv2d(hidden_dim, dim, 1)
        #将注意力机制的输出映射回输入维度
        self.to_out = nn.Linear(hidden_dim, dim, bias = False)

    def forward(
        self,
        x,
        pos_bias = None,
    ):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # split out heads
        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h = self.heads)
        if exists(self.rotary_emb):
            k = self.rotary_emb.rotate_queries_or_keys(k)
        # scale
        q = q * self.scale
        # rotate positions into queries and keys for time attention
        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
        # similarity
        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)#计算相似度
        # relative positional bias
        if exists(pos_bias):
            sim = sim + pos_bias
        # numerical stability
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)
        # aggregate values
        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)#使用注意力权重对值进行加权聚合
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)


class SignalEmbedding(nn.Module):
    def __init__(self, cond_arch, init_channel, channel_upsamplings):
        #channel_upsampling一个表示每个层的通道数变化的列表比如[16, 32, 64]
        super().__init__()
        if cond_arch == 'CNN':#构建卷积嵌入模型,对信号进行下采样和特征提取
            scale_factor = [init_channel, *map(lambda m: 1 * m, channel_upsamplings)]#定义缩放因子
            in_out_channels = list(zip(scale_factor[:-1], scale_factor[1:]))#生成输入输出通道对
            self.num_resolutions = len(in_out_channels)
            self.emb_model = self.generate_conv_embedding(in_out_channels)#生成卷积嵌入模型
        elif cond_arch == 'GRU':#构建 GRU 嵌入模型,进行序列数据的特征提取
            self.emb_model = nn.GRU(input_size = init_channel, hidden_size = channel_upsamplings[-1], num_layers = 3, batch_first=True)
        else:
            raise ValueError('Unknown architecture: {}'.format(cond_arch))

        self.cond_arch = cond_arch

    def Downsample1D(self, dim, dim_out = None):
        return nn.Conv1d(dim,default(dim_out, dim),kernel_size=4, stride=2, padding=1)
        #返回一个 nn.Conv1d 层，用于在一维信号上进行下采样（每次步长为 2）
    def generate_conv_embedding(self, channel_upsamplings):
        embedding_modules = nn.ModuleList([])
        for idx, (ch_in, ch_out) in enumerate(channel_upsamplings):
            embedding_modules.append(self.Downsample1D(ch_in,ch_out))
            embedding_modules.append(nn.SiLU())
        return nn.Sequential(*embedding_modules)
        #生成卷积嵌入模型，使用 Downsample1D 方法和 SiLU 激活函数
    def forward(self, x):
        # add channel dimension for conv1d
        if len(x.shape) == 2 and self.cond_arch == 'CNN':
            x = x.unsqueeze(1)
            x = self.emb_model(x)
        elif len(x.shape) == 2 and self.cond_arch == 'GRU':
            x = x.unsqueeze(2)
            x, _ = self.emb_model(x)
        x = torch.squeeze(x)
        return x

#不用的填充方式决定抽样的方法
#因为在上采样时，特别是对于周期性数据，可能需要特别的处理来正确地扩展和维持边界条件。但是下采样就不需要。
def Upsample(dim, padding_mode = 'zeros'):
    if padding_mode == 'zeros':
        return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1), padding_mode='zeros')
    elif padding_mode == 'circular':
        return CircularUpsample(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))
    else:
        raise ValueError('Unknown padding mode: {}'.format(padding_mode))
class CircularUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1):
        super(CircularUpsample, self).__init__()
        assert kernel_size[0] == 1 and kernel_size[1] == 4 and kernel_size[2] == 4
        assert stride[0] == 1 and stride[1] == 2 and stride[2] == 2
        assert padding[0] == 0 and padding[1] == 1 and padding[2] == 1
        assert dilation == 1
        if not isinstance(dilation, tuple):
            dilation = (dilation, dilation, dilation)
        self.true_padding = (dilation[0] * (kernel_size[0] - 1) - padding[0],
                             dilation[1] * (kernel_size[1] - 1) - padding[1],
                             dilation[2] * (kernel_size[2] - 1) - padding[2])
        # this ensures that no padding is applied by the ConvTranspose3d layer since we manually apply it before
        self.removed_padding = (dilation[0] * (kernel_size[0] - 1) + stride[0] + padding[0] - 1,
                             dilation[1] * (kernel_size[1] - 1) + stride[1] + padding[1] - 1,
                             dilation[2] * (kernel_size[2] - 1) + stride[2] + padding[2] - 1)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding=self.removed_padding)

    def forward(self, x):
        true_padding_repeated = tuple(i for i in reversed(self.true_padding) for _ in range(2))
        x = nn.functional.pad(x, true_padding_repeated, mode = 'circular') # manually apply padding of 1 on all sides
        x = self.conv_transpose(x)
        return x

def Downsample(dim, padding_mode='zeros'):
    if padding_mode == 'zeros' or padding_mode == 'circular':
        return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1), padding_mode=padding_mode)
    else:
        raise ValueError('Unknown padding mode: {}'.format(padding_mode))


class UNet(nn.Module):
    def __init__(
            self,
            dim,#基础维度，用于定义网络中各层的通道数
            channels=2,
            padding_mode='zeros',

            cond_to_time='add',
            dim_mults=(1, 2, 4),#用于在网络的不同层中逐步增加或调整特征维度的比例因子
            attn_heads=8,  # 注意力头的数量
            attn_dim_head=32,
            out_dim=None,
            self_condition=False, # 是否使用自我条件化（对于条件输入的双倍通道处理）
            init_dim = None,  # 初始卷积层的输出维度
            init_kernel_size = 5,
            resnet_groups=2,
            use_sparse_linear_attn=True,
            sigmoid_last_channel=False
    ):
        super().__init__()
        self.input_channels = channels * (2 if self_condition else 1)# 动态决定输入通道的数量，
        # 作用：1、多模态输入（如图像，文本）2、条件生成：模型的输入通道数可能会根据条件（如标签或附加信息）进行调整，以更好地适应条件生成任务。
        time_dim = dim * 4

        self.cond_dim = time_dim
        self.cond_to_time = cond_to_time
        self.padding_mode = padding_mode
        self.self_condition=self_condition



        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))
        # 旋转位置编码通过在自注意力机制中引入相对位置编码，改善了长序列的处理效果
        temporal_attn = lambda dim: EinopsToAndFrom('b c f h w', 'b (h w) f c',
                                                    Attention(dim, heads=attn_heads, dim_head=attn_dim_head,
                                                              rotary_emb=rotary_emb, cond_dim=self.cond_dim))
        # 是一个函数生成器，用于创建一个 EinopsToAndFrom 对象，该对象将指定的张量形状转换应用于注意力机制
        self.time_rel_pos_bias = RelativePositionBias(heads=attn_heads, max_distance=32)
        # 计算相对位置偏置，在上面有定义
        init_dim = default(init_dim, dim)

        assert is_odd(init_kernel_size)#奇数尺寸的内核可以在计算卷积时保持对称性
        init_padding = init_kernel_size // 2#计算在卷积操作中需要的填充（padding）大小，以保持输出尺寸与输入尺寸相同。
        if self.padding_mode == 'zeros' or self.padding_mode == 'circular':
            self.init_conv = nn.Conv3d(self.input_channels, init_dim, (1, init_kernel_size, init_kernel_size),
                                       padding=(0, init_padding, init_padding), padding_mode=self.padding_mode)
        else:
            raise ValueError('Unknown padding mode: {}'.format(self.padding_mode))
        # padding_mode 定义了在进行卷积操作时如何填充边界区域的像素。填充是卷积神经网络中非常重要的一个步骤，它决定了边界像素在卷积过程中如何处理

        self.init_temporal_attn = Residual(PreNorm(init_dim, temporal_attn(init_dim)))
        #PreNorm 在注意力机制之前对数据进行归一化，有助于稳定训练过程并提高模型的收敛性。
        #Residual 是一个残差模块，用于将输入和经过注意力机制处理后的输出进行相加。改善模型的训练过程。增强深度网络的表示能力
        # dimensions
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        ##根据指定的特征维度增长模式 (dim_mults) 生成网络中每一层的输入和输出维度

        # time conditioning
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )#一个用于时间条件的多层感知机（MLP），它可以用来处理与时间相关的信息或嵌入

        # CNN signal embedding for cond bias
        self.sign_emb_CNN = SignalEmbedding('CNN', init_channel=1, channel_upsamplings=(16, 32, 64, 128, self.cond_dim))
        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # block type
        block_klass = partial(ResnetBlock, padding_mode=self.padding_mode, groups=resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim=time_dim + int(
            self.cond_dim or 0) if self.cond_to_time == 'concat' else self.cond_dim)
        # partial函数允许你创建一个新的函数，该函数有一些参数预设为给定值，从而简化了函数调用

        # modules for all layers
        for ind, (dim_in, dim_out) in enumerate(in_out):  # 减少特征图的空间分辨率
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                block_klass_cond(dim_out, dim_out),
                Residual(PreNorm(dim_out, SpatialLinearAttention(dim_out, heads=attn_heads,
                                                                 cond_dim=self.cond_dim))) if use_sparse_linear_attn else nn.Identity(),
                Downsample(dim_out, self.padding_mode) if not is_last else nn.Identity()
            ]))  # 如果当前层不是最后一层则进行下采样。Downsample 模块用于减少特征图的空间维度。
        '''构建一个逐层的网络模块列表，并将这些模块按顺序添加到 self.downs 中。
        这个列表中的每一项都包含了一系列的网络层，这些层在输入数据流经它们时，
        会进行一系列的处理，包括卷积、残差连接、注意力机制、以及下采样等'''
        mid_dim = dims[-1]  # mid_dim将用于配置中间层的模块。
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)

        spatial_attn = EinopsToAndFrom('b c f h w', 'b f (h w) c',
                                       Attention(mid_dim, heads=attn_heads, cond_dim=self.cond_dim))
        # 这是一个 EinopsToAndFrom 实例，将 Attention 应用于重排列后的特征图，处理空间上的注意力
        # 创建残差模块
        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(PreNorm(mid_dim, temporal_attn(mid_dim)))
        # PreNorm: 在应用注意力机制之前对输入进行标准化

        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_in),
                block_klass_cond(dim_in, dim_in),
                Residual(PreNorm(dim_in, SpatialLinearAttention(dim_in, heads=attn_heads,
                                                                cond_dim=self.cond_dim))) if use_sparse_linear_attn else nn.Identity(),
                Upsample(dim_in, self.padding_mode) if not is_last else nn.Identity()
            ]))
        # 与down的模块相似，这些模块用于将特征图的分辨率逐步提高（作用相反）

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, out_dim, 1)
        )
        # 用于将网络的输出特征图转换为目标输出维度 out_dim

        # gradient embedding as in 'A physics-informed diffusion model for high-fidelity flow field reconstruction'
        self.emb_conv = nn.Sequential(
            torch.nn.Conv2d(channels, init_dim, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            torch.nn.Conv2d(init_dim, init_dim, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        )  # 对输入特征图进行初步的处理和特征提取
        self.combine_conv = torch.nn.Conv2d(init_dim * 2, init_dim, kernel_size=1, stride=1, padding=0)
        # 该卷积层用于将合并后的特征图通道数从 init_dim * 2 转换为 init_dim。
        # 这通常用于在特征图合并之后，调整通道数以便于进一步处理

        self.sigmoid_last_channel = sigmoid_last_channel  # 决定是否在模型的最后一层应用 Sigmoid 函数

    def forward(
        self,
        x,
        time,
        x_self_cond = None,
        cond = None,
        null_cond_prob = 0.
    ):
        batch, device = x.shape[0], x.device
        # reshape x to video-like input (since this U-Net is designed for video)
        video_flag = False
        if len(x.shape) == 3:
            x = generalized_b_xy_c_to_image(x)
            x = x.unsqueeze(2)
        elif len(x.shape) == 4:
            x = x.unsqueeze(2)
        elif len(x.shape) == 5:
            video_flag = True
        else:
            raise ValueError('Input must be image [BxCxPxP] or image sequence [BxCxFxPxP].')



        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)

        if exists(cond):
            '''在分类器自由引导的背景下，处理条件数据并将其与输入特征融合'''
            # classifier free guidance
            batch, device = x.shape[0], x.device
            mask = prob_mask_like((batch,), null_cond_prob, device=device)

            if len(cond.shape) == 3:
                label_mask_embed = rearrange(mask, 'b -> b 1 1')
                null_cond = torch.zeros_like(cond)
                cond = torch.where(label_mask_embed, null_cond, cond)  # cond 中的部分值替换为 null_cond。这在分类器自由引导中用于控制条件的应用。
                cond = generalized_b_xy_c_to_image(cond)
                cond_emb = self.emb_conv(cond)
                cond = cond
            else:
                raise ValueError('Input must be [BxP*PxC].')

            x = torch.cat((x.squeeze(2), cond_emb), dim=1)  # concatenate to channel dimension
            # 特征融合：将处理后的条件嵌入与模型输入进行融合，以结合条件信息进行后续处理
            x = self.combine_conv(x).unsqueeze(2)
            # 维度调整：确保输入和条件嵌入的维度适配模型的要求，并通过卷积层调整特征图的通道数

        r = x.clone()
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        for block1, block2, spatial_attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, spatial_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_conv(x)

        # # reshape to image if we have image-like data as input
        # if not video_flag:
        #     x = x.squeeze(2)
        # #这是一段处理视频和图像的转换，我们不涉及视频因此这一块不用

        if self.sigmoid_last_channel:
            # NOTE apply sigmoid on last channel of x to force E-field to be in [0,1]
            x[:, -1] = torch.sigmoid(x[:, -1])
        # 实现了 U-Net 的完整前向传播过程，
        # 包括处理图像/视频输入、条件信息、下采样、上采样、注意力机制以及生成最终输出
        return x.squeeze(2)

if __name__ == '__main__':
    model=UNet(dim=32,channels=1,sigmoid_last_channel=False).to('cpu')
    noisy=torch.randn(16,1,16,16).requires_grad_(True)

    # batch, pixel_x*pixel_y, channels，
    # 这里我们只要一个输出通道就可以，同时如果输入的是一维的向量，可以将一维的变成二维的数，只是别忘记再变换回来
    a=model(noisy,torch.tensor([1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.]))
    print(a.shape)