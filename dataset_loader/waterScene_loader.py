from pyquaternion import Quaternion

from dataset_loader.base_loader import BaseLoader
import torch
import numpy as np


class WaterScene_Loader(BaseLoader):
    """暂时只适配 WaterScene 数据集、全部点云作为 SearchArea， template box 没有偏移，适配 PGNN 算法，template 和 searchArea 随机采样。
    """
    def __init__(self, data, cfg):
        super(WaterScene_Loader, self).__init__(data, cfg)
        self.fullArea = cfg.full_area
        self.templateSize = cfg.template_size
        self.searchSize = cfg.search_size
        self.temp_offset = cfg.template_offset
        self.search_offset = cfg.search_area_offset

    def __len__(self):
        return self.data.num_frames

    def __getitem__(self, idx):
        """
        Returns:
            Dict {
                "template": Tensor[D1, P1],
                "boxCloud": Tensor[D2, P1],
                "searchArea": Tensor[D1, P2],
                "segLabel": [1, P2],
                "trueBox": Box
            }
        """
        trajID = torch.randint(0, self.data.num_trajectory, size=(1,)).item()
        framesNum = self.data.num_frames_trajectory(trajID)
        chosenIDs = torch.multinomial(torch.ones(framesNum), num_samples=2).tolist()
        tempFrame, searchFrame = self.data.frames(trajID, chosenIDs)

        tempOffset = np.random.uniform(low=-self.temp_offset, high=self.temp_offset, size=3)
        tempOffset[2] = tempOffset[2] * np.deg2rad(5)
        tempBox = tempFrame['3d_bbox']
        tempBox.translate(np.array([tempOffset[0], tempOffset[1], 0]))
        tempBox.rotate(Quaternion(axis=[0, 0, 1], radians=tempOffset[-1]))

        templatePoints = tempFrame['pc']
        template, _ = templatePoints.points_in_box(tempFrame['3d_bbox'], returnMask=False)
        template, _ = template.regularize(self.templateSize)
        boxCloud = template.box_cloud(tempFrame['3d_bbox'])
        # normTemplate = template.normalize()

        searchArea = searchFrame['pc']
        trueBox = searchFrame['3d_bbox']

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

