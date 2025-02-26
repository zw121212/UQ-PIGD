
import torch
from torch.utils.data import Dataset, DataLoader
class PairedDataset(Dataset):
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, idx):
        # 返回一对数据
        return self.data1[idx], self.data2[idx]

class PairedDataset2(Dataset):
    def __init__(self, data1, data2,data3):
        self.data1 = data1
        self.data2 = data2
        self.data3 = data3

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, idx):
        # 返回一对数据
        return self.data1[idx], self.data2[idx],self.data3[idx]
def create_f_s(x, y, w=0.125, r=10.):
    # 数用于生成一个具有特定条件下的权重矩阵或图像，
    # 值是根据输入坐标 (x, y) 和一些参数 w 和 r 来计算的
    condition1 = torch.abs(x - 0.5 * w) <= 0.5 * w
    condition2 = torch.abs(x - 1 + 0.5 * w) <= 0.5 * w
    condition3 = torch.abs(y - 0.5 * w) <= 0.5 * w
    condition4 = torch.abs(y - 1 + 0.5 * w) <= 0.5 * w

    result = torch.zeros_like(x)
    result[torch.logical_and(condition1, condition3)] = r
    result[torch.logical_and(condition2, condition4)] = -r
    return result

def create_f(pixels_per_dim,w=0.125,r=10.0,domain_size = 1.):
    # create stationary source field
    pixel_size = domain_size / pixels_per_dim
    start = pixel_size / 2
    end = domain_size - pixel_size / 2
    x = torch.linspace(start, end, steps=pixels_per_dim)
    y = torch.linspace(start, end, steps=pixels_per_dim)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    # compute the function values on the grid
    f_s = create_f_s(X, Y, w, r) # [pixels_per_dim, pixels_per_dim]
    return f_s

if __name__ == "__main__":
    a=create_f(64)
    print(a.shape)