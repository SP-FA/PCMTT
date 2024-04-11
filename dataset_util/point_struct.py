import copy
import torch
import numpy as np
import pandas as pd
from pyquaternion import Quaternion
import torch.nn.functional as F


class BasePointCloud:
    def __init__(self, points, dim):
        """
        Args:
            points (np.array[dim, n]) # dim 是点云维度
        """
        if dim != points.shape[0]:
            raise ValueError(f"{points.shape = } not matched at dim 0, which need {dim}")

        self.points = points
        self._dim = dim

    def __repr__(self):
        return f"{self.__class__}  shape: [{self.dim}, {self.n}]"

    @property
    def dim(self):
        return self._dim

    @property
    def n(self):
        return self.points.shape[1]

    def regularize(self, size):
        """使用随机采样法标准化。
        Args:
            size (int)

        TODO: 使用其他采样方法
        """
        selectIDs = None
        if self.n > 2:
            if self.n != size:
                selectIDs = np.random.choice(self.n, size=size, replace=size > self.n)
            else:
                selectIDs = np.arange(self.n)
        if selectIDs is not None:
            newPoints = self.points[:, selectIDs]
        else:
            newPoints = np.zeros((self._dim, size), dtype='float32')
        dataName = self.__class__
        points = dataName(newPoints)
        return points, selectIDs

    def box_cloud(self, box):
        """generate the BoxCloud for the given pc and box
        Returns:
            [N, 9]: The distance between each point and the box points
        """
        corner = box.corners()  # [3, 8]
        center = box.center.reshape(-1, 1)  # [3, 1]
        boxPoints = np.concatenate([center, corner], axis=1)  # [3, 9]
        pp = self.points[:3, :].T  # [3, n]
        pp = torch.tensor(pp, dtype=torch.float)
        boxPoints = boxPoints.T  # [3, 9]
        boxPoints = torch.tensor(boxPoints, dtype=torch.float)
        return torch.cdist(pp, boxPoints)

    def crop_points(self, maxi, mini, offset=None, returnMask=False):
        """
        Return:
            PointCloud
        """
        if offset is None:
            offset = [0, 0, 0]

        x1 = self.points[0, :] <= maxi[0] + offset[0]
        x2 = self.points[0, :] >= mini[0] - offset[0]
        y1 = self.points[1, :] <= maxi[1] + offset[1]
        y2 = self.points[1, :] >= mini[1] - offset[1]
        z1 = self.points[2, :] <= maxi[2] + offset[2]
        z2 = self.points[2, :] >= mini[2] - offset[2]

        includeIDs = np.logical_and(x1, x2)
        includeIDs = np.logical_and(includeIDs, y1)
        includeIDs = np.logical_and(includeIDs, y2)
        includeIDs = np.logical_and(includeIDs, z1)
        includeIDs = np.logical_and(includeIDs, z2)

        dataName = self.__class__
        cropped = dataName(self.points[:, includeIDs])
        if returnMask:
            return cropped, includeIDs
        return cropped

    def points_in_box(self, box, offset=None, returnMask=False, center=False):
        """给定一个 Bounding box，返回在这个 box 内的点

        Args:
            box (Box): Bounding box
            offset (float): 在 box 的 wlh 基础上额外增加的偏移量
            returnMask (bool): 是否返回哪些点在 box 内的 mask
            center (bool): 是否使点的中心在原点处

        Returns:
            WaterScene_PointCloud | KITTI_PointCloud: 在 box 内的点云，未进行 normalize
            Tensor[N, 9]: box cloud 向量，对于每个点，返回相对于 box 8 个 corner + 中心点 的距离
            Optional[Tensor[N]]: 返回一个 bool 向量，在 box 内的 point id 为 true
        """
        if offset is None:
            offset = [0, 0, 0]

        if returnMask is False:
            maxi = box.center + np.max(box.wlh)
            mini = box.center - np.max(box.wlh)
            newPoints = self.crop_points(maxi, mini, offset)
        else:
            newPoints = copy.deepcopy(self)

        newBox = copy.deepcopy(box)
        rotMat = box.rotation_matrix
        trans = box.center

        newBox.translate(-trans)
        newPoints.translate(-trans)
        newBox.rotate(Quaternion(matrix=(np.transpose(rotMat))))
        newPoints.rotate(np.transpose(rotMat))

        maxi =  newBox.wlh / 2
        mini = -newBox.wlh / 2
        pointInBox, includeIDs = newPoints.crop_points(maxi, mini, offset, True)

        if not center:
            pointInBox.rotate(rotMat)
            pointInBox.translate(trans)

        if returnMask:
            return pointInBox, pointInBox.box_cloud(box), includeIDs
        return pointInBox, pointInBox.box_cloud(box)

    def normalize(self):
        return F.normalize(self.convert2Tensor(), dim=-1)

    def convert2Tensor(self):
        tensorPoint = torch.from_numpy(self.points)
        tensorPoint = tensorPoint.float()
        return tensorPoint

    @classmethod
    def fromTensor(cls, tensor):
        points = tensor.numpy()
        return cls(points, points.shape[0])

    def translate(self, x):
        for i in range(3):
            self.points[i, :] = self.points[i, :] + x[i]

    def rotate(self, rot_mat):
        self.points[:3, :] = np.dot(rot_mat, self.points[:3, :])

    def transform(self, trans_mat):
        # 点之间距离可能发生变化
        self.points[:3, :] = trans_mat.dot(
            np.vstack(
                (self.points[:3, :], np.ones(self.n()))
            )
        )


class WaterScene_PointCloud(BasePointCloud):
    """适用于 WaterScene 数据集的点云数据结构
    """
    def __init__(self, points):
        """
        Args:
            points (np.array[dim, n]) # dim=8 是点云维度 [x, y, z, range, azimuth, elevation, rcs, doppler]
        """
        super().__init__(points, 8)

    @classmethod
    def from_file(cls, fileName):
        if fileName.endswith(".csv"):
            points = cls.load_csv(fileName)
        else:
            raise ValueError(f"Unsupported filetype {fileName}. Only supported .pcd file")
        return cls(points)

    @staticmethod
    def load_csv(fileName):
        df = pd.read_csv(fileName)
        df = df[["x", "y", "z", "range", "azimuth", "elevation", "rcs", "doppler"]]
        points = df.to_numpy()

        x = copy.deepcopy(points[:, 0])
        y = copy.deepcopy(points[:, 1])
        z = copy.deepcopy(points[:, 2])
        points[:, 0] = -z
        points[:, 1] =  x
        points[:, 2] = -y
        points = points.transpose()
        return points.reshape((8, -1))
    # TODO: 归一化


class KITTI_PointCloud(BasePointCloud):
    """适用于 KITTI 数据集的点云数据结构
    """

    def __init__(self, points):
        """
        Args:
            points (np.array[dim, n]) # dim 是点云维度, dim = 4
        """
        if points.shape[0] > 3:
            points = points[0:3, :]

        super().__init__(points, 3)

    @classmethod
    def from_file(cls, fileName):
        if fileName.endswith(".bin"):
            points = cls.load_pcd_bin(fileName)
        else:
            raise ValueError(f"Unsupported filetype {fileName}. Only supported .bin file")
        return cls(points)

    @staticmethod
    def load_pcd_bin(fileName):
        """从 binary 文件中读取数据。存储结构为：
           (x, y, z, intensity, ring, index)

        Returns:
            <np.array[4, n], np.float>
        """
        scan = np.fromfile(fileName, dtype=np.float32)
        points = scan.reshape((-1, 4))[:, :4]
        return points.T