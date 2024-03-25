"""
pointnet2.py
Created by zenn at 2021/5/9 13:41
"""

import torch
import torch.nn as nn

from model.pointnet2.utils.pointnet2_modules import PointnetSAModule


class Pointnet2(nn.Module):
    r"""
        PointNet2 with single-scale grouping
        Semantic segmentation network that uses feature propogation layers
    """

    def __init__(self, use_fps=False, normalize_xyz=False, return_intermediate=False, input_channels=0):
        """
        Args:
            use_fps: 是否使用 fps 采样
            normalize_xyz
            return_intermediate: 是否返回所有结果
            input_channels: 除了 xyz 之外的维度数量
        """
        super(Pointnet2, self).__init__()
        self.return_intermediate = return_intermediate
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                radius=0.3,
                nsample=32,
                mlp=[input_channels, 64, 64, 128],
                use_xyz=True,
                use_fps=use_fps, normalize_xyz=normalize_xyz
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                radius=0.5,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
                use_fps=False, normalize_xyz=normalize_xyz
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                radius=0.7,
                nsample=32,
                mlp=[256, 256, 256, 256],
                use_xyz=True,
                use_fps=False, normalize_xyz=normalize_xyz
            )
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, pointcloud, numpoints):
        """
        Args:
            pointcloud: 点云，前三个维度必须为 x, y, z，后面是特征
            numpoints List[int * 3]: 三层 PointNet++ 分别下采样到多少点
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features, l_idxs = [xyz], [features], []
        for i in range(len(self.SA_modules)):
            li_xyz, li_features, sample_idxs = self.SA_modules[i](l_xyz[i], l_features[i], numpoints[i], True)
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            l_idxs.append(sample_idxs)
        if self.return_intermediate:
            return l_xyz[1:], l_features[1:], l_idxs[0]
        return l_xyz[-1], l_features[-1], l_idxs[0]
