from torch import nn
from model.attention.dual_attention import DualAttention


class TemplateFeatExtraction(nn.Module):
    def __init__(self, in_channels=256 + 9, out_channels=256):
        super(TemplateFeatExtraction, self).__init__()
        # TODO: DGN

        self.mlp = nn.Conv1d(in_channels, out_channels, kernel_size=1)  # [B, D2 + D3, P1] -> [B, D2, P1]
        self.att = DualAttention(out_channels, out_channels)

    def forward(self, template, tempbox):
        """Input a template
        Args:
            x Tensor[B, P1, D2 + D3]

        Returns:
            Tensor[B, D2, P1]
        """

        # TODO: template -> DGN


        x = x.permute(0, 2, 1)

        x = self.mlp(x)
        return self.att(x)

