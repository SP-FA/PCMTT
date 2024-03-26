import dataset_util.kitti
from base_loader import BaseLoader
# class KITTI_Loader(BaseLoader):
# def __init__(self, totalArea=True): #-> template, searchArea, box
import torch
from dataset_util.data_struct import KITTI_PointCloud
import numpy as np



class KITTI_Loader(BaseLoader):
    def __init__(self, kitti_util , template_size, search_size, sigma=1.0,data=None):
        self.kitti_util = kitti_util
        super(KITTI_Loader, self).__init__(data)
        self.template_size = template_size
        self.search_size = search_size
        self.sigma = sigma

    def __len__(self):
        return self.kitti_util.num_scenes

    def __getitem__(self, idx):
        scene_id = self.kitti_util.scenes_list[idx]
        scene_data = self.kitti_util.frames(scene_id)
        target_box = scene_data["3d_bbox"]

        template_area = self.generate_template_area(target_box, self.template_size)
        search_area = self._generate_search_area(target_box, self.search_size)
        #获取得分
        template_score = self.getScoreGaussian(target_box.center, self.sigma)
        search_score = self.getScoreGaussian(target_box.center, self.sigma)
        #转为tensor类型
        template_area_tensor = self._point_cloud_to_tensor(template_area)
        search_area_tensor = self._point_cloud_to_tensor(search_area)
        template_score_tensor = torch.tensor([template_score], dtype=torch.float32)
        search_score_tensor = torch.tensor([search_score], dtype=torch.float32)
        box_tensor = target_box.to_tensor()

        return template_area_tensor, search_area_tensor, template_score_tensor,search_score_tensor, box_tensor

    def generate_template_area(self, box, size):

        points = self.kitti_util.frames["pc"].points
        bbox = box.to_bbox()
        template_center = np.array([bbox[0], bbox[2]])
        template_extent = max(bbox[3], bbox[4], bbox[5]) * size
        template_area = KITTI_PointCloud(points)
        for point in points:
            if (template_center[0] - template_extent <= point[0] <= template_center[0] + template_extent and
                    template_center[2] - template_extent <= point[2] <= template_center[2] + template_extent):
                template_area.points.append(point)
        return template_area

    def _generate_search_area(self, box, size):

        points = self.kitti_util.frames["pc"].points
        bbox = box.to_bbox()
        search_center = np.array([bbox[0], bbox[2]])
        search_extent = max(bbox[3], bbox[4], bbox[5]) * size
        search_area = KITTI_PointCloud()
        for point in points:
            if (point[0] >= search_center[0] - search_extent and
                    point[0] <= search_center[0] + search_extent and
                    point[2] >= search_center[2] - search_extent and
                    point[2] <= search_center[2] + search_extent):
                search_area.points.append(point)
        return search_area

    def _point_cloud_to_tensor(self, point_cloud):

        points_array = np.array(point_cloud.points).astype(np.float32)
        points_tensor = torch.from_numpy(points_array).float()
        return points_tensor

    def getScoreGaussian(self, center, sigma):

        distance = np.linalg.norm(center)
        return np.exp(-distance ** 2 / (2 * sigma ** 2))


kitti_util = dataset_util.kitti.KITTI_Util(r"C:\Users\49465\Desktop\dataset_mini\kitti_mini", "traintiny", coordinate_mode="velodyne", preloading=True)

# 然后，使用 KITTI_Util 实例作为参数来创建 KITTI_Loader 实例
loader = KITTI_Loader(kitti_util=kitti_util, template_size=2.0, search_size=2.0, sigma=1.0,data=None)
print("ok")