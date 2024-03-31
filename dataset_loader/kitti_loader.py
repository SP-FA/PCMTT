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
        self.temp_offset = cfg.template_offset
        self.search_offset = cfg.search_area_offset

    def __len__(self):
        return self.data.num_scenes

    def __getitem__(self, idx):
        trajID = torch.randint(0, self.data.num_trajectory, size=(1,)).item()
        framesNum = self.data.num_frames_trajectory(trajID)
        chosenIDs = torch.multinomial(torch.ones(framesNum), num_samples=2).tolist()
        tempFrame, searchFrame = self.data.frames(trajID, chosenIDs)

        tempOffset = np.random.uniform(low=-self.temp_offset, high=self.temp_offset, size=4)
        tempOffset[3] = tempOffset[3] * np.deg2rad(5)
        tempOffset[2] = 0
        tempBox = tempFrame['3d_bbox']
        tempBox.translate(tempOffset[:-1])
        tempBox.rotate(Quaternion(axis=[0, 0, 1], radians=tempOffset[-1]))

        templatePoints = tempFrame['pc']
        template, _ = templatePoints.points_in_box(tempBox, returnMask=False)
        template, _ = template.regularize(self.templateSize)
        boxCloud = template.box_cloud(tempFrame['3d_bbox'])
        normTemplate = template.normalize()

        searchPoints = searchFrame['pc']
        trueBox = searchFrame['3d_bbox']
        if self.fullArea:
            searchPoints, _ = searchPoints.regularize(self.searchSize)
            _, _, segLabel = searchPoints.points_in_box(trueBox, returnMask=True)
            searchArea = searchPoints
        else:
            searchOffset = [self.search_offset, self.search_offset, self.search_offset]
            searchArea, _, segLabel = searchPoints.points_in_box(trueBox,offset=searchOffset, returnMask=True)
            searchArea, selectIDs = searchArea.regularize(self.searchSize)
            segLabel = segLabel[selectIDs]
        return {
            "template": normTemplate.to(self.cfg.device),
            "boxCloud": boxCloud.clone().detach().to(self.cfg.device),
            "searchArea": searchArea.convert2Tensor().to(self.cfg.device),
            "segLabel": torch.tensor(segLabel).to(self.cfg.device),
            "trueBox": torch.tensor([trueBox.center[0], trueBox.center[1], trueBox.center[2],
                        trueBox.wlh[0], trueBox.wlh[1], trueBox.wlh[2], trueBox.theta]).view(-1).to(self.cfg.device)
        }
