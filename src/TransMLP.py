import torch.nn as nn
import torch.nn.functional as F

# 定义一个残差块
class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

        # 使用He初始化
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += residual  # 残差连接
        return F.relu(out)

class TransMLP_fu(nn.Module):
    def __init__(self, input_dim,  output_dim, num_res_blocks,num_layers=10):
        super(TransMLP_fu, self).__init__()
        hidden_dim=input_dim
        self.input_fc = nn.Linear(input_dim , hidden_dim)
        self.bn_input = nn.BatchNorm1d(hidden_dim)

        # 使用He初始化
        nn.init.kaiming_normal_(self.input_fc.weight)

        # 构建多个残差块
        self.res_blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_res_blocks)])
        self.extra_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ) for _ in range(num_layers)
        ])

        self.output_fc = nn.Linear(hidden_dim, output_dim)
        self.bn_output = nn.BatchNorm1d(output_dim)

        # 使用He初始化
        nn.init.kaiming_normal_(self.output_fc.weight)

    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()  # [batch ,channel ,size]

        x = x.view(batch_size * seq_len, input_dim)
        x = F.relu(self.bn_input(self.input_fc(x)))
        for block in self.res_blocks:
            x = block(x)
        for layer in self.extra_layers:
            x = layer(x)

        x = self.bn_output(self.output_fc(x))

        x = x.view(batch_size, seq_len, -1)
        return x
    # def forward_with_highdimension(model,x):
    #     num_matrices,batch_size,channel,size = x.size()
    #     outputs = []
    #     for i in range(num_matrices):
    #         x_i = x[i, :, :, :]
    #         out_i = model(x_i)
    #         outputs.append(out_i)
    #     return torch.stack(outputs, dim=0)

# if __name__ == '__main__':
#     input_dim = 256
#     output_dim = 256
#     import time
#     from src.Unet1D import *
#     start_time = time.time()
#     model = TransMLP_fu(input_dim,  output_dim, num_res_blocks=5)
#     x1=torch.randn(32,1,16,16)
#     x1=x1.reshape(32,1,256)
#
#     out1=model(x1)
#     end_time = time.time()
#     print('time:',end_time-start_time)
#     print("Output1 shape:", out1.shape)
