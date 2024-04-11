import torch
from torch import nn

from model.DGN.dgn import DGN
from model.attention.dual_attention import DualAttention
from model.pointnet2.pointnet2 import Pointnet2


class SearchAreaFeatExtraction(nn.Module):
    def __init__(self, cfg, pn2, dgn, in_channel):
        super(SearchAreaFeatExtraction, self).__init__()
        self.cfg = cfg
        if cfg.dataset.lower() == "waterscene":
            input_channels = 5
        else:
            input_channels = 0
        self.pn2 = pn2
        self.dgn = dgn
        # self.pn2 = Pointnet2(cfg.use_fps, cfg.normalize_xyz, input_channels)  # [B, 256, P3]
        # self.dgn = DGN(in_channel)
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
        assert torch.any(torch.isnan(feat)) == torch.tensor(False)
        return xyz, self.att(feat), sample_idxs  # [B, 256, P3]

        # output = self.dgn(x)  # [B, P2, D1]
        # output = output.permute(0, 2, 1)
        # xyz = x[:, :3, :]
        # return xyz.permute(0, 2, 1), self.att(output), torch.tensor(list(range(x.shape[-1]))).to(self.cfg.device)
