import torch
from torch import nn

from model.DGN.dgn import DGN
from model.attention.dual_attention import DualAttention
from model.pointnet2.pointnet2 import Pointnet2


class SearchAreaFeatExtraction(nn.Module):
    def __init__(self, backbone, in_channel, cfg):
        super(SearchAreaFeatExtraction, self).__init__()
        self.cfg = cfg
        if cfg.dataset.lower() == "waterscene":
            input_channels = 5
        else:
            input_channels = 0
        self.backbone = backbone
        self.att = DualAttention(256, 256)

    def forward(self, x):
        """Input a search area
        Args:
            x Tensor[B, D1, P2]

        Returns:
            Tensor[B, D2=256, P2]
        """
        if self.cfg.backbone.lower() == "dgn":
            xyz = x[:, :3, :]
            x = self.backbone(x)  # [B, D1, P2]
            x = x.permute(0, 2, 1)
            return xyz.permute(0, 2, 1), self.att(x, xyz), torch.tensor(list(range(x.shape[-1]))).repeat(x.shape[0], 1).to(self.cfg.device)
        else:
            N = x.shape[-1]
            x = x.permute(0, 2, 1)
            xyz, feat, sample_idxs = self.backbone(x, [N // 2, N // 4, N // 8])  # feature: [B, 256, P3]
            return xyz, self.att(feat, xyz), sample_idxs  # [B, 256, P3]
