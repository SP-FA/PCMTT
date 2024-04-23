import copy

from pyquaternion import Quaternion

from dataset_loader import points_utils
from dataset_loader.base_loader import BaseLoader
import torch
import numpy as np

from dataset_util.point_struct import KITTI_PointCloud


class KITTI_Loader(BaseLoader):
    """暂时只适配 kitti 数据集，适配 PGNN 算法，template 和 searchArea 随机采样。
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
        """
        return {
            "template": tensor[P1, D1],
            "boxCloud": tensor,
            "searchArea": tensor[P2, D1],
            "segLabel": tensor,
            "trueBox": tensor[delta center * 3, wlh * 3, delta degree]
        }
        """
        trajID = torch.randint(0, self.data.num_trajectory, size=(1,)).item()
        framesNum = self.data.num_frames_trajectory(trajID)
        chosenIDs = torch.multinomial(torch.ones(framesNum), num_samples=2).tolist()
        tempFrame, searchFrame = self.data.frames(trajID, chosenIDs)

        # get temp box
        rng = self.cfg.rand_distortion_range
        tempOffset = np.random.uniform(low=-rng, high=rng, size=3)
        tempOffset[2] = tempOffset[2] * 5
        tempBox = tempFrame['3d_bbox']
        tempBox = tempBox.get_offset_box(tempOffset, self.cfg.box_enlarge_scale)

        # get template
        templatePoints = tempFrame['pc']
        template, _ = templatePoints.points_in_box(tempBox, returnMask=False, center=True)
        template, _ = template.regularize(self.templateSize)
        # normTemplate = template.normalize()

        # get box cloud
        boxCloud = template.box_cloud(tempBox)  # TODO: 这个有 Bug，要改

        # get search area & true box
        searchArea = searchFrame['pc']
        trueBox = searchFrame['3d_bbox']
        searchOffset = self.gaussian.sample(1)[0]
        sampleBox = copy.deepcopy(trueBox)
        sampleBox = sampleBox.get_offset_box(searchOffset, self.cfg.box_enlarge_scale, limit=True)

        if self.cfg.full_area:
            searchArea.translate(-sampleBox.center)
            searchArea.rotate(np.transpose(sampleBox.rotation_matrix))
        else:
            searchArea, _ = searchArea.points_in_box(sampleBox, [self.offset, self.offset, self.offset], center=True)

        trueBox.translate(-sampleBox.center)
        trueBox.rotate(Quaternion(matrix=sampleBox.rotation_matrix.T))

        searchArea, _ = searchArea.regularize(self.searchSize)
        _, _, segLabel = searchArea.points_in_box(trueBox, returnMask=True, center=True)

        return {
            "template": template.convert2Tensor(),
            "boxCloud": boxCloud.clone().detach(),
            "searchArea": searchArea.convert2Tensor(),
            "segLabel": torch.tensor(segLabel).float(),
            "trueBox": torch.tensor([trueBox.center[0], trueBox.center[1], trueBox.center[2],
                                     trueBox.wlh[0], trueBox.wlh[1], trueBox.wlh[2], -searchOffset[2]]).view(-1)
        }
