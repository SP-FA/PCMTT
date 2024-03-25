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
        temp = data["template"]
        area = data["searchArea"]
        box  = data["templateBox"]

        tf = self.tempFeat(temp, box)
        af = self.areaFeat(area)
        predBox, predCls, vote_xyz, center_xyz, finalBox = self.joinFeat(tf, af)
        # return predBox, predCls, vote_xyz, center_xyz
        return {
            "predBox": predBox,
            "predCls": predCls,
            "vote_xyz": vote_xyz,
            "center_xyz": center_xyz,
            "finalBox": finalBox
        }

    def training_step(self, batch, batch_idx):
        """
        Args:
            batch: {
                "template": Tensor[B, D1, P1]
                "searchArea": Tensor[B, D1, P2]
                "templateBox": Tensor[B, D3, P1]
                "labelBox": List[Box * B]
                "trackID": List[int * B]
            }
        """
        res = self(batch)
        predCls = res['predCls']  # B,N
        N = predCls.shape[1]
        seg_label = batch['seg_label']
        sample_idxs = res['sample_idxs']  # B,N
        # update label
        seg_label = seg_label.gather(dim=1, index=sample_idxs[:, :N].long())  # B,N
        batch["seg_label"] = seg_label
        # compute loss
        loss_dict = self.compute_loss(batch, res)
        loss = loss_dict['loss_objective'] * self.config.objectiveness_weight \
               + loss_dict['loss_box'] * self.config.box_weight \
               + loss_dict['loss_seg'] * self.config.seg_weight \
               + loss_dict['loss_vote'] * self.config.vote_weight
        return loss
