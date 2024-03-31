from torch import nn
from model.attention.dual_attention import DualAttention
from model.pointnet2.pointnet2 import Pointnet2


class SearchAreaFeatExtraction(nn.Module):
    def __init__(self, cfg):
        super(SearchAreaFeatExtraction, self).__init__()
        self.config = cfg
        if cfg.dataset.lower() == "waterscene":
            input_channels = 5
        else:
            input_channels = 0
        self.pn2 = Pointnet2(cfg.use_fps, cfg.normalize_xyz, input_channels)  # [B, 256, P3]
        self.att = DualAttention(256, 256)

    def forward(self, x):
        """Input a search area
        Args:
            x Tensor[B, D1, P2]

        Returns:
            Tensor[B, D2=256, P2]
        """
        N = x.shape[-1]
        x = x.permute(0, 2, 1)
        xyz, feat, sample_idxs = self.pn2(x, [N // 2, N // 4, N // 8])  # feature: [B, 256, P3]
        return xyz, self.att(feat), sample_idxs  # [B, 256, P3]


