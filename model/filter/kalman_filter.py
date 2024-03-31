import torch
import torch.nn as nn


class KalmanFilter(nn.Module):
    """可学习的 Kalman filter
    Attributes:
        A Tensor[3, 3]: 转换矩阵（状态转移矩阵）
        B float: delta t
        H Tensor[3, 3]: 观测矩阵（测量值权值矩阵）
        Q float: 过程噪声方差
        R float: 测量噪声方差
        x0 Tensor[3, 1]: x, y, z
        P0 Tensor[3, 3]: 先验估计协方差
    """
    def __init__(self):

        super(KalmanFilter, self).__init__()
        self.A = nn.Parameter(torch.eye(3), requires_grad=False)
        self.B = nn.Parameter(torch.eye(3) * 0.01)
        self.H = nn.Parameter(torch.eye(3), requires_grad=False)
        self.Q = nn.Parameter(torch.tensor([0.1]))
        self.R = nn.Parameter(torch.tensor([1], dtype=torch.float))
        self.x = None
        self.P = nn.Parameter(torch.eye(3), requires_grad=False)

    def predict(self, u):
        """
        Args:
            u Tensor[3, 1]: 速度向量，可以用 bbox 内点的 doppler 均值
        """
        self.x = torch.mm(self.A, self.x) + torch.mm(self.B, u)
        self.P = torch.mm(torch.mm(self.A, self.P), self.A.t()) + self.Q
        return self.x

    def update(self, z):
        """
        Args:
            z Tensor[3, 1]: MMT 的预测值
        """
        if self.x is None:
            self.x = z
        else:
            y = z - torch.mm(self.H, self.x)
            S = self.R + torch.mm(torch.mm(self.H, self.P), self.H.t())
            K = torch.mm(torch.mm(self.P, self.H.t()), torch.inv(S))
            self.x = self.x + torch.mm(K, y)
            I = torch.eye(self.x.size(0), dtype=self.x.dtype)
            self.P = (torch.mm(torch.mm(I - torch.mm(K, self.H), self.P), (I - torch.mm(K, self.H)).t()) +
                      torch.mm(torch.mm(K, self.R), K.t()))

    def forward(self, u, z):
        if self.x is not None:
            output = self.predict(u)
        else:
            output = z

        self.update(z)
        return output
