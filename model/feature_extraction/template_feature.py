import torch
from torch import nn

from model.DGN.dgn import DGN
from model.attention.dual_attention import DualAttention


class TemplateFeatExtraction(nn.Module):
    def __init__(self, in_channel, out_channel=256):
        super(TemplateFeatExtraction, self).__init__()
        # TODO: DGN

        self.dgn = DGN(in_channel)
        self.mlp = nn.Conv1d(256 + 9, out_channel, kernel_size=1)  # [B, D2 + D3, P1] -> [B, D2, P1]
        self.att = DualAttention(out_channel, out_channel)

    def forward(self, template, boxCloud):
        """Input templates and boxClouds
        Args:
            template Tensor[B, D1, P1]
            boxCloud Tensor[B, P1, D3]

        Returns:
            Tensor[B, D2, P1]
        """
        template = self.dgn(template)  # [B, P1, D2]
        x = torch.cat([template, boxCloud], dim=-1)  # [B, P1, D2 + D3]
        x = x.permute(0, 2, 1)
        x = self.mlp(x)
        return self.att(x)

