import torch
import torch.nn as nn
class ResidualCNNNetwork(nn.Module):
    def __init__(self, size, in_channels=1, hidden_channels=16):
        super(ResidualCNNNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # 添加通道维度：[batch, 1, size, size]
        x = x.unsqueeze(1)
        residual = x  # 残差
        out = self.cnn(x)
        out += residual  # 加入残差
        return out.squeeze(1)  # 去掉通道维度

if __name__ == '__main__':
    model = ResidualCNNNetwork(size=64)
    x=torch.randn(16,64,64)
    print(model(x).shape)