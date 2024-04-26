import torch
import torch.nn as nn

from model.DGN.dgn import DGN
from model.feature_extraction.search_area_feature import SearchAreaFeatExtraction
from model.feature_extraction.template_feature import TemplateFeatExtraction
from model.feature_fusion import FeatureFusion
from model.pointnet2.pointnet2 import Pointnet2
from model.repSurf.umbrella import RepSurfUmbrella


class PGNN(nn.Module):
    def __init__(self, cfg):
        super(PGNN, self).__init__()
        self.device = cfg.device
        self.useRepSurf = cfg.use_repSurf
        if cfg.dataset.lower() == "waterscene":
            if cfg.use_repSurf:
                dim = 9
            else:
                dim = 8
        elif cfg.dataset.lower() == "kitti":
            if cfg.use_repSurf:
                dim = 9
            else:
                dim = 3

        if cfg.dataset.lower() == "waterscene":
            if cfg.use_repSurf:
                input_channels = 6
            else:
                input_channels = 5
        else:
            if cfg.use_repSurf:
                input_channels = 6
            else:
                input_channels = 0

        if cfg.use_repSurf:
            self.repSurf = RepSurfUmbrella(cfg=cfg)

        if cfg.backbone.lower() == "dgn":
            self.backbone = DGN(dim)
        else:
            self.backbone = Pointnet2(cfg.use_fps, cfg.normalize_xyz, input_channels)  # [B, 256, P3]

        self.tempFeat = TemplateFeatExtraction(self.backbone, dim, cfg).to(cfg.device)
        self.areaFeat = SearchAreaFeatExtraction(self.backbone, dim, cfg).to(cfg.device)
        self.joinFeat = FeatureFusion(256, cfg).to(cfg.device)

    def forward(self, data):
        temp = data["template"].to(self.device)
        box = data["boxCloud"].to(self.device)
        area = data["searchArea"].to(self.device)

        if self.usrRepSurf:
            temp = self.repSurf(temp)
            area = self.repSurf(area)

        tf = self.tempFeat(temp, box)
        # tf += 1e-6
        # assert torch.any(torch.isnan(tf)) == torch.tensor(False)
        xyz, af, sample_idxs = self.areaFeat(area)
        # assert torch.any(torch.isnan(af)) == torch.tensor(False)
        res = self.joinFeat(tf, af, xyz)
        return res, sample_idxs
