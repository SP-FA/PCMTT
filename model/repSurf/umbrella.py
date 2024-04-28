import torch
import torch.nn as nn

from model.DGN.knn import get_graph_feature
from model.repSurf.coordinate_util import cartesian2polar, cal_normal, cal_center, cal_constant, check_nan


class RepSurfUmbrella(nn.Module):
    """
    Umbrella-based Surface Abstraction Module

    """
    def __init__(self, k=9, aggregation='sum', includeDist=False, randInvNormal=True, cfg=None):
        """
        Args:
            k (int): 选取附近的 k 个邻居
            in_channel (int):
            aggregation (str: {'max', 'avg', 'sum'}): 特征增强方式
            includeDist (bool): 特征是否包含平面的距离（平面中心点 点乘 法向量，可以获得平面与原点的距离）
            randInvNormal (bool): 随机旋转法向量方向
        """
        super(RepSurfUmbrella, self).__init__()
        self.k = k + 1
        self.includeDist = includeDist
        self.randInvNormal = randInvNormal
        self.aggregation = aggregation
        self.cfg = cfg
        in_channel = 10 if includeDist else 9

        self.mlps = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channel, in_channel, 1, bias=True),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channel, in_channel, 1, bias=True),
        ).to(cfg.device)

    def forward(self, x):
        """
        Args:
            x (tensor): [B, C, N]

        Returns:
            tensor: [B, C, N]
        """
        xyz = x[:, :3, :]
        groupPoints = self.group_by_umbrella(xyz, self.k)
        groupNormal = cal_normal(groupPoints, self.randInvNormal)  # [B, N, k, 3]
        groupCenter = cal_center(groupPoints)  # [B, N, k, 3]
        groupPolar  = cartesian2polar(groupCenter)  # [B, N, k, 3]

        if self.includeDist:
            groupPos = cal_constant(groupNormal, groupCenter)
            groupNormal, groupCenter, groupPos = check_nan(groupNormal, groupCenter, groupPos)
            feature = torch.cat([groupCenter, groupPolar, groupNormal, groupPos], dim=-1)  # [B, N, k, 10]
        else:
            groupNormal, groupCenter = check_nan(groupNormal, groupCenter)
            feature = torch.cat([groupCenter, groupPolar, groupNormal], dim=-1)  # [B, N, k, 9]
        feature = feature.permute(0, 3, 2, 1)  # [B, C, k, N]
        feature = self.mlps(feature)

        if self.aggregation == 'max':   feature = torch.max(feature, 2)[0]
        elif self.aggregation == 'avg': feature = torch.mean(feature, 2)
        else:                           feature = torch.sum(feature, 2)
        return feature

    def group_by_umbrella(self, xyz, k):
        """
        Args:
            xyz (tensor): [B, 3, N]
            k (int)

        Return:
            [B, N, k-1, 3 (points), 3 (coord.)]
        """
        groupPoint = get_graph_feature(xyz, k)[:, :, 1:]  # [B, N, k-1, 3]
        torch.cuda.empty_cache()
        xyz = xyz.permute(0, 2, 1)  # [B, N, 3]
        groupPointNorm = groupPoint - xyz.unsqueeze(-2)
        polarPoints = cartesian2polar(groupPointNorm)  # [B, N, k-1, 3]
        phi = polarPoints[..., 2]
        idx = phi.argsort(dim=-1)

        B, N, k, _ = groupPointNorm.shape
        idxB = torch.arange(B, dtype=torch.long).to(self.cfg.device).view(B, 1, 1).repeat(1, N, k)
        idxN = torch.arange(N, dtype=torch.long).to(self.cfg.device).view(1, N, 1).repeat(B, 1, k)
        sortGroupPointNorm = groupPointNorm[idxB, idxN, idx, :].unsqueeze(-2)
        sortGroupPointNormRoll = torch.roll(sortGroupPointNorm, -1, dims=-3)
        centriod = torch.zeros_like(sortGroupPointNorm)
        return torch.cat([centriod, sortGroupPointNorm, sortGroupPointNormRoll], dim=-2)

