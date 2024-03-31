import torch
import torch.nn.functional as F
import torch.nn as nn

from model.feature_extraction.search_area_feature import SearchAreaFeatExtraction
from model.feature_extraction.template_feature import TemplateFeatExtraction
from model.feature_fusion import FeatureFusion


class PGNN(nn.Module):
    def __init__(self, cfg):
        super(PGNN, self).__init__()
        self.device = cfg.device
        if cfg.dataset.lower() == "waterscene":
            dim = 8
        elif cfg.dataset.lower() == "kitti":
            dim = 3

        self.tempFeat = TemplateFeatExtraction(dim).to(cfg.device)
        self.areaFeat = SearchAreaFeatExtraction(cfg).to(cfg.device)
        self.joinFeat = FeatureFusion(256, cfg).to(cfg.device)

    def forward(self, data):
        temp = data["template"]
        box = data["boxCloud"]
        area = data["searchArea"]

        tf = self.tempFeat(temp, box)
        xyz, af, sample_idxs = self.areaFeat(area)
        res = self.joinFeat(tf, af, xyz)
        return res, sample_idxs

    # def training_step(self, batch, batch_idx):
    #     """
    #     Args:
    #         batch: {
    #             "template": Tensor[B, D1, P1]
    #             "boxCloud": Tensor[B, D3, P1]
    #             "searchArea": Tensor[B, D1, P2]
    #             "segLabel": List[Box * B]
    #             "trueBox": List[B * Box]
    #         }
    #     """
    #     res = self(batch)
    #     # predSeg = res['predSeg']  # B,N
    #     # N = predSeg.shape[1]
    #     # seg_label = batch['seg_label']
    #     # sample_idxs = res['sample_idxs']  # B,N
    #     # update label
    #     # seg_label = seg_label.gather(dim=1, index=sample_idxs[:, :N].long())  # B,N
    #     # batch["seg_label"] = seg_label
    #     # compute loss
    #     loss_dict = self.compute_loss(batch, res)
    #     loss = loss_dict['loss_objective'] * self.cfg.objective_weight + \
    #            loss_dict['loss_box'] * self.cfg.box_weight + \
    #            loss_dict['loss_seg'] * self.cfg.seg_weight + \
    #            loss_dict['loss_vote'] * self.cfg.vote_weight
    #     return loss
