import torch.nn as nn
from model.attention.dual_attention import PositionAttention


class CrossAttention(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm1d):
        super(CrossAttention, self).__init__()
        inter_channels = in_channels // 4
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv1d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.sa = PositionAttention(inter_channels)
        self.conv3 = nn.Sequential(nn.Conv1d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.conv4 = nn.Sequential(nn.Dropout1d(0.1, False), nn.Conv1d(inter_channels, out_channels, 1))

    def forward(self, x1, x2):
        """
        Args:
            x1 Tensor[B, D2, P1]: Template
            x2 Tensor[B, D2, P2]: SearchArea

        Returns:
            Tensor[B, D2, P2]
        """
        feat1 = self.conv1(x1)
        feat2 = self.conv2(x2)
        sa_feat = self.sa(feat1, feat2)
        sa_conv = self.conv3(sa_feat)
        return self.conv4(sa_conv)
