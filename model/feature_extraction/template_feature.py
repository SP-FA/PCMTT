import torch
from torch import nn


class TemplateFeatExtraction(nn.Module):
    def __init__(self, backbone, att, cfg, out_channel=256):
        super(TemplateFeatExtraction, self).__init__()
        self.backbone = backbone
        self.cfg = cfg

        self.mlp = nn.Conv1d(256 + 9, out_channel, kernel_size=1)  # [B, D2 + D3, P1] -> [B, D2, P1]
        self.att = att

    def forward(self, x, boxCloud):
        """Input templates and boxClouds
        Args:
            template Tensor[B, D1, P1]
            boxCloud Tensor[B, P1, D3]

        Returns:
            Tensor[B, D2, P1]
        """
        if self.cfg.backbone.lower() == "dgn":
            x = self.backbone(x)  # [B, D2, P1]
            x = x.permute(0, 2, 1)
            return self.att(x)
        else:
            N = x.shape[-1]
            x = x.permute(0, 2, 1)
            xyz, feat, sample_idxs = self.backbone(x, [N // 2, N // 4, N // 8])  # feature: [B, 256, P1]
            return self.att(feat)

        # 拼接 box cloud
        # x = torch.cat([x, boxCloud], dim=-1)  # [B, P1, D2 + D3]
        # x = self.mlp(x)
