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


def criterion(batch, output, sample_idxs):
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
    segLabel = batch['segLabel'].to(cfg.device)
    trueBox = batch['trueBox'].to(cfg.device)  # B,4
    center_xyz = output["center_xyz"]  # B,num_proposal,3
    vote_xyz = output["vote_xyz"]

    # N = predSeg.shape[1]
    # segLabel = segLabel.gather(dim=1, index=sample_idxs[:, :N].long())

    loss_seg = F.binary_cross_entropy_with_logits(predSeg, segLabel)

    loss_vote = F.smooth_l1_loss(vote_xyz, trueBox[:, None, :3].expand_as(vote_xyz), reduction='none')  # B,N,3
    loss_vote = (loss_vote.mean(2) * segLabel).sum() / (segLabel.sum() + 1e-06)

    dist = torch.sum((center_xyz - trueBox[:, None, :3]) ** 2, dim=-1)

    dist = torch.sqrt(dist + 1e-6)  # B, K
    objectness_label = torch.zeros_like(dist, dtype=torch.float)
    objectness_label[dist < 0.3] = 1
    objectness_score = predBox[:, :, 4]  # B, K
    objectness_mask = torch.zeros_like(objectness_label, dtype=torch.float)
    objectness_mask[dist < 0.3] = 1
    objectness_mask[dist > 0.6] = 1
    loss_objective = F.binary_cross_entropy_with_logits(objectness_score, objectness_label,
                                                        pos_weight=torch.tensor([2.0]).cuda())
    loss_objective = torch.sum(loss_objective * objectness_mask) / (
            torch.sum(objectness_mask) + 1e-6)
    loss_box = F.smooth_l1_loss(predBox[:, :, :4],
                                trueBox[:, None, :4].expand_as(predBox[:, :, :4]),
                                reduction='none')
    loss_box = torch.sum(loss_box.mean(2) * objectness_label) / (objectness_label.sum() + 1e-6)

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
        optim = SGD(model.parameters(), lr=cfg.lr, momentum=0.9)
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
                loss = criterion(batch, res, sample_idxs)

                optim.zero_grad()
                loss.backward()
                optim.step()
                totalLoss += loss

                fb = res['finalBox'][0]
                tb = batch['trueBox'][0]
                tb = tb.cpu().detach().numpy()
                fb = fb.cpu().detach().numpy()
                trueBox    = Box((tb[0], tb[1], tb[2]), (tb[3], tb[4], tb[5]), tb[6], Quaternion(axis=[0, 0, 1], degrees=tb[6]))
                pridictBox = Box((fb[0], fb[1], fb[2]), (tb[3], tb[4], tb[5]), fb[3], Quaternion(axis=[0, 0, 1], degrees=fb[3]))
                #print(f"true: {trueBox}")
                #print(f"      {pridictBox}")

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
                    loss = criterion(batch, res, sample_idxs)
                    validLoss += loss

                fb = res['finalBox'][0]
                tb = batch['trueBox'][0]
                tb = tb.cpu().detach().numpy()
                fb = fb.cpu().detach().numpy()
                trueBox    = Box((tb[0], tb[1], tb[2]), (tb[3], tb[4], tb[5]), tb[6], Quaternion(axis=[0, 0, 1], degrees=tb[6]))
                pridictBox = Box((fb[0], fb[1], fb[2]), (tb[3], tb[4], tb[5]), fb[3], Quaternion(axis=[0, 0, 1], degrees=fb[3]))

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
