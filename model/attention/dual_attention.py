import torch
from torch import nn


class PositionEmbedding(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel=3, num_pos_feats=256):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        """
        Args:
            xyz (tensor): [B, 3, N]
        """
        return self.embedding(xyz)


class AddNorm(nn.Module):
    def __init__(self, dim):
        super(AddNorm, self).__init__()
        self.dim = dim

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input, x):
        res = self.dropout(input) + x
        shapeList = res.shape
        res = res.view(-1, self.dim)
        res = self.norm(res)
        return res.view(shapeList)


class DualAttention(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm1d):
        super(DualAttention, self).__init__()
        self.embed = PositionEmbedding()
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
        self.addnorm_sa1 = AddNorm(inter_channels)
        self.addnorm_sa2 = AddNorm(out_channels)
        self.addnorm_sc1 = AddNorm(inter_channels)
        self.addnorm_sc2 = AddNorm(out_channels)

    def forward(self, x, xyz):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_feat = self.addnorm_sa1(sa_feat, feat1)
        sa_conv = self.conv51(sa_feat)
        # sa_conv = self.addnorm_sa2(sa_conv, sa_feat)
        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_feat = self.addnorm_sc1(sc_feat, feat2)
        sc_conv = self.conv52(sc_feat)
        # sc_conv = self.addnorm_sc2(sc_conv, sc_feat)
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
            x1 Tensor[B, C, P1]: k, v, Template
            x2 Tensor[B, C, P2]: q, SearchArea
        """
        B, C, P1 = x1.size()
        k = self.K(x1).view(B, -1, P1)  # [B, C, P1]
        v = self.V(x1).view(B, -1, P1)  # [B, C, P1]
        if x2 is None:
            q = self.Q(x1).view(B, -1, P1).permute(0, 2, 1)  # [B, P1, C]
        else:
            B, C, P2 = x2.size()
            q = self.Q(x2).view(B, -1, P2).permute(0, 2, 1)  # [B, P2, C]

        energy = torch.bmm(q, k)  # [B, P1, P3]
        attention = self.softmax(energy)
        out = torch.bmm(v, attention.permute(0, 2, 1))  # [B, C, P2]

        if x2 is not None:
            return x2
        return x1


class ChannelAttention(nn.Module):
    def __init__(self, dim):
        super(ChannelAttention, self).__init__()
        self.chanel_in = dim

        self.Q = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.K = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.V = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1)

        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, P = x.size()
        q = self.Q(x).view(B, -1, P)  # [B, C, P1]
        v = self.V(x).view(B, -1, P)  # [B, C, P1]
        k = self.K(x).view(B, -1, P).permute(0, 2, 1)  # [B, P1, C]

        energy = torch.bmm(q, k)
        attention = self.softmax(energy)
        out = torch.bmm(attention, v)
        return x
