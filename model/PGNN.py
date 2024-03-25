import torch.nn as nn

from model.feature_extraction.search_area_feature import SearchAreaFeatExtraction
from model.feature_extraction.template_feature import TemplateFeatExtraction
from model.feature_fusion import FeatureFusion


class PGNN(nn.Module):
    def __init__(self, cfg):
        super(PGNN, self).__init__()
        self.tempFeat = TemplateFeatExtraction()
        self.areaFeat = SearchAreaFeatExtraction(cfg)
        self.joinFeat = FeatureFusion(256, cfg)

    def forward(self, data):
        temp, area, box = data
        tf = self.tempFeat(temp, box)
        af = self.areaFeat(area)
        jf = self.joinFeat(tf, af)
        # TODO: 处理 jf
