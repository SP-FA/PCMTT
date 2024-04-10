import torch
import torch.nn as nn
import torch.nn.functional as F
from model.DGN.knn import get_graph_feature


class DGN(nn.Module):
    def __init__(self, in_channel, out_channel=256, k=5):
        super(DGN, self).__init__()
        self.k = k
        self.edgeConv1 = EdgeConv(in_channel, 64, k)
        self.edgeConv2 = EdgeConv(64, 64, k)
        self.edgeConv3 = EdgeConv(64, 64, k)

        self.conv = nn.Conv1d(3 * 64, 1024, kernel_size=1)
        self.mlp1 = nn.Sequential(
            nn.Linear(3 * 64 + 1024, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5)
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5)
        )
        self.mlp4 = nn.Linear(128, out_channel)

    def forward(self, x):  # [B, N, C]
        """
        Args:
            x Tensor[B, C, N]: 归一化过的点云
        """
        N = x.shape[1]
        x1 = self.edgeConv1(x)
        x2 = self.edgeConv2(x1)
        x3 = self.edgeConv3(x2)
        x = torch.cat([x1, x2, x3], dim=-1)  # [B, N, 3 * 64]
        x4 = x.permute(0, 2, 1)

        globalFeat = self.conv(x4)  # [B, 1024, N]
        globalFeat = globalFeat.max(dim=-1, keepdim=False)[0]  # [B, 1024]
        globalFeat = globalFeat.unsqueeze(1).repeat(1, N, 1)  # [B, N, 1024]
        x = torch.cat([x, globalFeat], dim=-1)  # [B, N, 3 * 64 + 1024]

        x = self.mlp1(x)  # [B, N, 256]
        x = self.mlp2(x)  # [B, N, 256]
        x = self.mlp3(x)  # [B, N, 128]
        return self.mlp4(x)  # [B, N, out_channel]


class EdgeConv(nn.Module):
    def __init__(self, in_channel, out_channel, k):
        super(EdgeConv, self).__init__()
        self.k = k
        self.relu = nn.LeakyReLU()
        self.linear11 = nn.Linear(in_channel, 2 * in_channel)
        self.linear12 = nn.Linear(in_channel, 2 * in_channel)
        self.linear2  = nn.Linear(2 * in_channel, in_channel)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Args:
            x Tensor[B, C, N]: 归一化过的点云

        Returns:
            Tensor[B, N, C]
        """
        x_neighbour = get_graph_feature(x, self.k)  # [B, N, k, C]
        x = x.unsqueeze(2)
        x = self.linear11(x)  # [B, N, 1, 2 * C]
        x_neighbour = self.linear12(x_neighbour)  # [B, N, k, 2 * C]
        c = self.linear2(self.relu(x + x_neighbour))  # [B, N, k, C]
        x_neighbour = (x_neighbour - x) * c
        x_neighbour = x_neighbour.permute(0, 3, 1, 2)  # [B, C, N, k]
        x_neighbour = self.conv1(x_neighbour)  # [B, C, N, k]
        x = x.permute(0, 3, 1, 2)
        x = self.conv2(x)  # [B, C, N, 1]
        x = self.conv3(x)
        x = x + x_neighbour
        return x.max(dim=-1, keepdim=False)[0].transpose(1, 2)  # [B, N, C]