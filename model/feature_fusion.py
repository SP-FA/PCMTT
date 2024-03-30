import torch
import torch.nn as nn
from model.attention.cross_attention import CrossAttention
from model.filter.kalman_filter import KalmanFilter
from model.vote_net.rpn import RPN


class FeatureFusion(nn.Module):
    def __init__(self, in_channels, cfg):
        super(FeatureFusion, self).__init__()
        self.att = CrossAttention(in_channels, in_channels)
        self.kal = KalmanFilter()
        self.rpn = RPN(in_channels, cfg.normalize_xyz)
        self.prevB = None
        self.prevV = None

    def forward(self, x1, x2):
        """
        Args:
            x1 Tensor[B, D2, P1]: Template
            x2 Tensor[B, D2, P2]: SearchArea

        Returns:
            {
            "predBox": Tensor[B, 4+1, num_proposal]
            "predSeg": Tensor[B, N]
            "vote_xyz": Tensor[B, N, 3]
            "center_xyz": Tensor[B, num_proposal, 3]
            "finalBox": [B, 4]
        }
        """
        output = self.att(x1, x2)  # [B, D2, P2]
        xyz = x2[:, :3, :]  # [B, 3, P2]
        # if self.prevB is not None:
            # ...
            # 用 kalman 预先把 xyz 剪裁一下，但是不知道会不会导致模型难以收敛，暂时先不启用
            # TODO: PRN with Kalman

        # [B, P2, 4+1], [B, P2]
        predBox, predSeg, vote_xyz, center_xyz = self.rpn(xyz, output)
        print(f"{predBox.shape = }  {predSeg.shape = }  {vote_xyz.shape = }  {center_xyz.shape = }")

        # bestBoxID = torch.argmax(predBox[:, 4])  # [B, 1]
        # bestBox = predBox[bestBoxID]
        # bestBoxCenter = center_xyz[bestBoxID]
        # velo = self.get_velocity(xyz, bestBox)  # [B, 1]
        # TODO: predict and update kalman
        # TODO: get the final result with (prevB + output) / 2

        return {
            "predBox": predBox,
            "predSeg": predSeg,
            "vote_xyz": vote_xyz,
            "center_xyz": center_xyz,
            # "finalBox":
        }
