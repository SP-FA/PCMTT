import unittest
from pyquaternion import Quaternion
from dataset_util.point_struct import KITTI_PointCloud


class MyTestCase(unittest.TestCase):
    def test_KITTI_PointCloud(self):
        p = KITTI_PointCloud.from_file(
            "H:/E_extension/dataset/kitti/data_tracking/velodyne/training/velodyne/0000/000000.bin")
        print(p.points.shape)
        print(p.n)

    def test_KITTI_PointCloud_affine(self):
        p = KITTI_PointCloud.from_file(
            "H:/E_extension/dataset/kitti/data_tracking/velodyne/training/velodyne/0000/000000.bin")
        print(p.points)
        p.translate([200, 200, 200])
        print(p.points)

        q = Quaternion()
        p.rotate(q.rotation_matrix)
        print(p.points)


if __name__ == '__main__':
    unittest.main()