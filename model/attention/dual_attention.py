import torch
from torch import nn


class DualAttention(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm1d):
        super(DualAttention, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv1d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv1d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.sa = PositionAttention(inter_channels)
        self.sc = ChannelAttention(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv1d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv1d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout1d(0.1, False), nn.Conv1d(inter_channels, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout1d(0.1, False), nn.Conv1d(inter_channels, out_channels, 1))
        self.conv8 = nn.Sequential(nn.Dropout1d(0.1, False), nn.Conv1d(inter_channels, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv + sc_conv
        sasc_output = self.conv8(feat_sum)
        return sasc_output


class PositionAttention(nn.Module):
    def __init__(self, dim):
        super(PositionAttention, self).__init__()
        self.chanel_in = dim

        self.Q = nn.Conv1d(in_channels=dim, out_channels=dim // 8, kernel_size=1)
        self.K = nn.Conv1d(in_channels=dim, out_channels=dim // 8, kernel_size=1)
        self.V = nn.Conv1d(in_channels=dim, out_channels=dim     , kernel_size=1)

        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2=None):
        """
        Args:
            x1 Tensor[B, C, P1]: q, v, Template
            x2 Tensor[B, C, P3]: k, SearchArea
        """
        B, C, P1 = x1.size()
        q = self.Q(x1).view(B, -1, P1).permute(0, 2, 1)  # [B, P1, C]
        v = self.V(x1).view(B, C, P1)  # [B, C, P1]
        if x2 is None:
            k = self.K(x1).view(B, -1, P1)
        else:
            B, C, P2 = x2.size()
            k = self.K(x2).view(B, -1, P2)  # [B, C, P3]

        energy = torch.bmm(q, k)  # [B, P1, P3]
        attention = self.softmax(energy)

        if x2 is None:
            out = torch.bmm(v, attention.permute(0, 2, 1))  # [B, C, P1]
            out = out.view(B, C, P1)
        else:
            out = torch.bmm(v, attention)  # [B, C, P3]
            out = out.view(B, C, P2)

        if x2 is not None:
            return self.alpha * out + x2
        return self.alpha * out + x1


class ChannelAttention(nn.Module):
    def __init__(self, dim):
        super(ChannelAttention, self).__init__()
        self.chanel_in = dim

        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, P = x.size()
        q = x.view(B, C, -1)
        k = x.view(B, C, -1).permute(0, 2, 1)
        v = x.view(B, C, -1)
        energy = torch.bmm(q, k)
        invEnergy = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy  # prevent loss divergence
        attention = self.softmax(invEnergy)

        out = torch.bmm(attention, v)
        out = out.view(B, C, P)
        return self.alpha * out + x
