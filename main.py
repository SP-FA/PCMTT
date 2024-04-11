import os

import yaml
import argparse
from easydict import EasyDict
import torch
from pyquaternion import Quaternion
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset_loader.kitti_loader import KITTI_Loader
from dataset_loader.waterScene_loader import WaterScene_Loader
from dataset_util.box_struct import Box
from dataset_util.kitti import KITTI_Util
from dataset_util.waterscene import WaterScene_Util
from model.PGNN import PGNN
from metrics import Success, Precision
from metrics import estimateOverlap, estimateAccuracy


def load_yaml(file_name):
    with open(file_name, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    return config


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default="./config/PGNN_WaterScene.yaml", help='the config_file')
    parser.add_argument('--test', action='store_true', default=False, help='test mode')
    parser.add_argument('--preloading', action='store_true', default=False, help='preload dataset into memory')
    parser.add_argument('--device', default="cuda:0")

    args = parser.parse_args()
    config = load_yaml(args.cfg)
    config.update(vars(args))
    return EasyDict(config)


def criterion(batch, output):
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
    segLabel = batch['segLabel']
    trueBox = batch['trueBox'].to(cfg.device)
    segLabel = batch['segLabel'].to(cfg.device)
    M = sample_idxs.shape[1]
    segLabel = segLabel.gather(dim=1, index=sample_idxs[:, :M].long())
    segLabel = segLabel.float()

    loss_seg = F.binary_cross_entropy_with_logits(predSeg, segLabel)
    assert torch.any(torch.isnan(predSeg)) == torch.tensor(False)
    assert torch.any(torch.isnan(loss_seg)) == torch.tensor(False)

    loss_vote = F.smooth_l1_loss(vote_xyz, trueBox[:, None, :3].expand_as(vote_xyz), reduction='none')
    loss_vote = (loss_vote.mean(2) * segLabel).sum() / (segLabel.sum() + 1e-06)
    assert torch.any(torch.isnan(loss_vote)) == torch.tensor(False)

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
    assert torch.any(torch.isnan(loss_objective)) == torch.tensor(False)

    # near_object_score = object_score[dist < 0.3]
    # far_object_score  = object_score[dist > 0.6]
    # far_object_score  = 1 - far_object_score
    # near_loss_objective = F.binary_cross_entropy_with_logits(near_object_score, object_label[dist < 0.3],
    #                                                     pos_weight=torch.tensor([2.0]).to(cfg.device))
    # far_loss_objective  = F.binary_cross_entropy_with_logits(far_object_score, object_label[dist > 0.6],
    #                                                     pos_weight=torch.tensor([2.0]).to(cfg.device))
    # loss_objective = near_loss_objective + far_loss_objective
    # assert torch.any(torch.isnan(near_loss_objective)) == torch.tensor(False)
    # assert torch.any(torch.isnan(far_loss_objective)) == torch.tensor(False)

    # loss_objective = torch.sum(loss_objective * object_mask) / (torch.sum(object_mask) + 1e-6)

    loss_box = F.smooth_l1_loss(predBox[:, :, :4],
                                trueBox[:, None, :4].expand_as(predBox[:, :, :4]),
                                reduction='none')
    loss_box = torch.sum(loss_box.mean(2) * object_label) / (object_label.sum() + 1e-6)
    assert torch.any(torch.isnan(loss_box)) == torch.tensor(False)

    totalLoss = loss_objective * cfg.object_weight + \
                loss_box * cfg.box_weight + \
                loss_seg * cfg.seg_weight + \
                loss_vote * cfg.vote_weight
    return totalLoss


if __name__ == "__main__":
    cfg = parse_config()
    print(cfg)

    cfg.device = torch.device(cfg.device)

    if cfg.pretrain is None:
        model = PGNN(cfg)
    else:
        model = torch.load(cfg.pretrain)

    if cfg.optimizer.lower() == 'sgd':
        optim = SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.wd)
    else:
        optim = Adam(model.parameters(), lr=cfg.lr, betas=(0.5, 0.999), eps=1e-06)  # , weight_decay=cfg.wd

    if cfg.dataset.lower() == "waterscene":
        trainData = WaterScene_Util(cfg, cfg.train_split)
        validData = WaterScene_Util(cfg, cfg.valid_split)
        train = WaterScene_Loader(trainData, cfg)
        valid = WaterScene_Loader(validData, cfg)
    elif cfg.dataset.lower() == "kitti":
        trainData = KITTI_Util(cfg, cfg.train_split)
        validData = KITTI_Util(cfg, cfg.valid_split)
        train = KITTI_Loader(trainData, cfg)
        valid = KITTI_Loader(validData, cfg)

    trainLoader = DataLoader(train, batch_size=cfg.batch_size, num_workers=cfg.workers, pin_memory=True, shuffle=True)
    validLoader = DataLoader(valid, batch_size=1             , num_workers=cfg.workers, pin_memory=True)

    if cfg.test is False:
        bestAcc = -1e9
        # train
        for i_epoch in range(cfg.epoch):
            totalLoss = 0
            Success_train = Success()
            Precision_train = Precision()

            model.train()
            for batch in trainLoader:
                res, sample_idxs = model(batch)
                loss = criterion(batch, res)

                optim.zero_grad()
                loss.backward()
                optim.step()
                totalLoss += loss

                fb = res['finalBox'][0]
                tb = batch['trueBox'][0]
                tb = tb.cpu().detach().numpy()
                fb = fb.cpu().detach().numpy()
                trueBox    = Box((tb[0], tb[1], tb[2]), (tb[3], tb[4], tb[5]), tb[6], Quaternion(axis=[0, 0, 1], radians=tb[6]))
                pridictBox = Box((fb[0], fb[1], fb[2]), (tb[3], tb[4], tb[5]), tb[6], Quaternion(axis=[0, 0, 1], radians=tb[6]))
                # print(f"true: {trueBox}")
                # print(pridictBox)

                # pridictBoxList.append(pridictBox)
                overlap = estimateOverlap(trueBox, pridictBox)
                accuracy = estimateAccuracy(trueBox, pridictBox)

                Success_train.add_overlap(overlap)
                Precision_train.add_accuracy(accuracy)

            print(f"{i_epoch} / {cfg.epoch}: loss = {totalLoss / len(trainLoader)}  total loss = {totalLoss}")
            print(f'\ttrain Succ/Prec: {Success_train.average:.1f}/{Precision_train.average:.1f}')

            Success_train.reset()
            Precision_train.reset()

            if cfg.save_last:
                torch.save(model, os.path.join(cfg.checkpoint, f"last_model-{i_epoch}.pt"))

            if i_epoch % cfg.check_val_every_n_epoch != 0:
                continue

            # valid
            Success_valid = Success()
            Precision_valid = Precision()

            model.eval()
            validLoss = 0
            for batch in validLoader:
                with torch.no_grad():
                    res, sample_idxs = model(batch)
                    loss = criterion(batch, res)
                    validLoss += loss

                fb = res['finalBox'][0]
                tb = batch['trueBox'][0]
                tb = tb.cpu().detach().numpy()
                fb = fb.cpu().detach().numpy()
                trueBox    = Box((tb[0], tb[1], tb[2]), (tb[3], tb[4], tb[5]), tb[6], Quaternion(axis=[0, 0, 1], radians=tb[6]))
                pridictBox = Box((fb[0], fb[1], fb[2]), (tb[3], tb[4], tb[5]), tb[6], Quaternion(axis=[0, 0, 1], radians=tb[6]))

                # pridictBoxList.append(pridictBox)
                overlap = estimateOverlap(trueBox, pridictBox)
                accuracy = estimateAccuracy(trueBox, pridictBox)

                Success_valid.add_overlap(overlap)
                Precision_valid.add_accuracy(accuracy)

            print(f'\t\t\t\t\tValid {i_epoch}: loss = {validLoss / len(validLoader)}  total loss = {validLoss}')
            print(f'\t\t\t\t\t\tvalid Succ/Prec: {Success_valid.average:.1f}/{Precision_valid.average:.1f}')

            if bestAcc < Precision_valid.average:
                bestAcc = Precision_valid.average
                torch.save(model, os.path.join(cfg.checkpoint, f"best_model-{i_epoch}-{Success_valid.average}-{Precision_valid.average}.pt"))

            Success_valid.reset()
            Precision_valid.reset()
    else:
        ...
