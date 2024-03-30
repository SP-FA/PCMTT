import torch
import torch.functional as F
import torch.nn as nn

from model.feature_extraction.search_area_feature import SearchAreaFeatExtraction
from model.feature_extraction.template_feature import TemplateFeatExtraction
from model.feature_fusion import FeatureFusion


class PGNN(nn.Module):
    def __init__(self, cfg):
        super(PGNN, self).__init__()
        self.cfg = cfg
        self.tempFeat = TemplateFeatExtraction()
        self.areaFeat = SearchAreaFeatExtraction(cfg)
        self.joinFeat = FeatureFusion(256, cfg)

    def forward(self, data):
        temp = data["template"]
        box = data["boxCloud"]
        area = data["searchArea"]

        tf = self.tempFeat(temp, box)
        af = self.areaFeat(area)
        predBox, predSeg, vote_xyz, center_xyz = self.joinFeat(tf, af)  # , finalBox
        return {
            "predBox": predBox,
            "predSeg": predSeg,
            "vote_xyz": vote_xyz,
            "center_xyz": center_xyz,
            # "finalBox": finalBox
        }

    def training_step(self, batch, batch_idx):
        """
        Args:
            batch: {
                "template": Tensor[B, D1, P1]
                "boxCloud": Tensor[B, D3, P1]
                "searchArea": Tensor[B, D1, P2]
                "segLabel": List[Box * B]
                "trueBox": List[B * Box]
            }
        """
        res = self(batch)
        # predSeg = res['predSeg']  # B,N
        # N = predSeg.shape[1]
        # seg_label = batch['seg_label']
        # sample_idxs = res['sample_idxs']  # B,N
        # update label
        # seg_label = seg_label.gather(dim=1, index=sample_idxs[:, :N].long())  # B,N
        # batch["seg_label"] = seg_label
        # compute loss
        loss_dict = self.compute_loss(batch, res)
        loss = loss_dict['loss_objective'] * self.cfg.objective_weight + \
               loss_dict['loss_box'] * self.cfg.box_weight + \
               loss_dict['loss_seg'] * self.cfg.seg_weight + \
               loss_dict['loss_vote'] * self.cfg.vote_weight
        return loss

    def compute_loss(self, data, output):
        """
        Args:
            batch: {
                "template": Tensor[B, D1, P1]
                "boxCloud": Tensor[B, D3, P1]
                "searchArea": Tensor[B, D1, P2]
                "segLabel": List[Box * B]
                "trueBox": List[B * Box]
            }
            output: {
                "predBox": Tensor[B, 4+1, num_proposal]
                "predSeg": Tensor[B, N]
                "vote_xyz": Tensor[B, N, 3]
                "center_xyz": Tensor[B, num_proposal, 3]
                "finalBox": [B, 4]
            }

        Returns:
            {
            "loss_objective": float
            "loss_box": float
            "loss_seg": float
            "loss_vote": float
        }
        """
        predBox = output['predBox']  # B,num_proposal,5
        predSeg = output['predSeg']  # B,N
        center_xyz = output["center_xyz"]  # B,num_proposal,3
        vote_xyz = output["vote_xyz"]
        segLabel = data['segLabel']
        trueBox = data['trueBox']  # B,4

        loss_seg = F.binary_cross_entropy_with_logits(predSeg, segLabel)

        loss_vote = F.smooth_l1_loss(vote_xyz, trueBox[:, None, :3].expand_as(vote_xyz), reduction='none')
        loss_vote = (loss_vote.mean(2) * segLabel).sum() / (segLabel.sum() + 1e-06)

        dist = torch.sum((center_xyz - trueBox[:, None, :3]) ** 2, dim=-1)
        dist = torch.sqrt(dist + 1e-6)  # B, K

        object_label = torch.zeros_like(dist, dtype=torch.float)
        object_label[dist < 0.3] = 1
        object_score = predBox[:, :, 4]  # B, K
        object_mask = torch.zeros_like(object_label, dtype=torch.float)
        object_mask[dist < 0.3] = 1
        object_mask[dist > 0.6] = 1
        loss_objective = F.binary_cross_entropy_with_logits(object_score, object_label,
                                                            pos_weight=torch.tensor([2.0]).cuda())
        loss_objective = torch.sum(loss_objective * object_mask) / (
                torch.sum(object_mask) + 1e-6)
        loss_box = F.smooth_l1_loss(predBox[:, :, :4],
                                    trueBox[:, None, :4].expand_as(predBox[:, :, :4]),
                                    reduction='none')
        loss_box = torch.sum(loss_box.mean(2) * object_label) / (object_label.sum() + 1e-6)

        return {
            "loss_objective": loss_objective,
            "loss_box": loss_box,
            "loss_seg": loss_seg,
            "loss_vote": loss_vote,
        }
