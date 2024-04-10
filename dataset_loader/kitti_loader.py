import copy

from pyquaternion import Quaternion

from dataset_loader.base_loader import BaseLoader
import torch
import numpy as np


class KITTI_Loader(BaseLoader):
    """暂时只适配 kitti 数据集、SearchArea 可选范围， template box 随机偏移，适配 PGNN 算法，template 和 searchArea 随机采样。
        """
    def __init__(self, data, cfg):
        super(KITTI_Loader, self).__init__(data, cfg)
        self.fullArea = cfg.full_area
        self.templateSize = cfg.template_size
        self.searchSize = cfg.search_size
        self.offset = cfg.search_area_offset

    def __len__(self):
        return self.data.num_frames

    def __getitem__(self, idx):
        trajID = torch.randint(0, self.data.num_trajectory, size=(1,)).item()
        framesNum = self.data.num_frames_trajectory(trajID)
        chosenIDs = torch.multinomial(torch.ones(framesNum), num_samples=2).tolist()
        tempFrame, searchFrame = self.data.frames(trajID, chosenIDs)

        # get temp box
        tempBox = tempFrame['3d_bbox']
        tempBox.scale(self.cfg.box_enlarge_scale)
        offset = self.cfg.rand_distortion_range
        tempOffset = np.random.uniform(low=-offset, high=offset, size=3)
        tempBox.translate(np.array([tempOffset[0], tempOffset[1], 0]))
        tempBox.rotate(Quaternion(axis=[0, 0, 1], radians=tempOffset[2] * np.deg2rad(5)))

        # get template
        templatePoints = tempFrame['pc']
        template, _ = templatePoints.points_in_box(tempBox, returnMask=False)
        template, _ = template.regularize(self.templateSize)
        # normTemplate = template.normalize()

        # get box cloud
        boxCloud = template.box_cloud(tempBox)

        # get search area & true box
        searchArea = searchFrame['pc']
        trueBox = searchFrame['3d_bbox']
        if self.cfg.full_area is False:
            searchArea, _ = searchArea.points_in_box(trueBox, [self.offset, self.offset, self.offset])

        searchArea, _ = searchArea.regularize(self.searchSize)
        _, _, segLabel = searchArea.points_in_box(trueBox, returnMask=True)

        return {
            "template": template.convert2Tensor(),
            "boxCloud": boxCloud.clone().detach(),
            "searchArea": searchArea.convert2Tensor(),
            "segLabel": torch.tensor(segLabel).float(),
            "trueBox": torch.tensor([trueBox.center[0], trueBox.center[1], trueBox.center[2],
                        trueBox.wlh[0], trueBox.wlh[1], trueBox.wlh[2], trueBox.theta]).view(-1)
        }
