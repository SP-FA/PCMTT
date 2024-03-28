import copy

import numpy as np
import pandas as pd

from dataset_util.base_class import BasePointCloud


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
        print(scan.shape)
        points = scan.reshape((-1, 4))[:, :4]
        return points.T
