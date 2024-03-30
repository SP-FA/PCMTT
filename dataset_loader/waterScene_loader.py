import torch
from dataset_loader.base_loader import BaseLoader


def random_choice(num_samples, size):
    return torch.multinomial(
        torch.ones(size, dtype=torch.float32),
        num_samples=num_samples
    )


class WaterScene_Loader(BaseLoader):
    """暂时只适配 WaterScene 数据集、全部点云作为 SearchArea， template box 没有偏移，适配 PGNN 算法，template 和 searchArea 随机采样。
    """
    def __init__(self, data, cfg):
        super(WaterScene_Loader, self).__init__(data)
        self.fullArea = cfg.full_area
        self.templateSize = cfg.template_size
        self.searchSize = cfg.search_size

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
        templateFrame, searchFrame = self.data.frames(trajID, chosenIDs)

        templatePoints = templateFrame['pc']
        template, _ = templatePoints.points_in_box(templateFrame['3d_bbox'], returnMask=False)
        template, _ = template.regularize(self.templateSize)
        boxCloud = template.box_cloud(templateFrame['3d_bbox'])
        template = template.normalize()

        searchAreaPoints = searchFrame['pc']
        trueBox = searchFrame['3d_bbox']
        if self.fullArea:
            searchAreaPoints, _ = searchAreaPoints.regularize(self.searchSize)
            _, _, segLabel = searchAreaPoints.points_in_box(trueBox, returnMask=True)
            searchArea = searchAreaPoints
        else:
            # TODO: 对 trueBox 进行处理
            searchArea, _, segLabel = searchAreaPoints.points_in_box(trueBox, returnMask=True)
            searchArea, selectIDs = searchArea.regularize(self.searchSize)
            segLabel = segLabel[selectIDs]
        return {
            "template": template,
            "boxCloud": boxCloud,
            "searchArea": searchArea,
            "segLabel": segLabel,
            "trueBox": trueBox
        }

