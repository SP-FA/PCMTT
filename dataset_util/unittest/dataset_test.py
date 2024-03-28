import unittest
from dataset_util.kitti import KITTI_Util
from dataset_util.waterscene import WaterScene_Util


class MyTestCase(unittest.TestCase):
    def test_kitti(self):
        kitti = KITTI_Util("H:/fyp/Open3DSOT/kitti", "train tiny", preloading=True)
        print(kitti.num_scenes)
        print(kitti.num_frames)
        print(kitti.num_trajectory)
        print(kitti.num_frames_trajectory(10))
        print(kitti.frames(0, [0]))

    def test_waterscene(self):
        water = WaterScene_Util("H:/E_extension/dataset/waterScene", "train tiny", preloading=False)
        print(water.num_scenes)
        print(water.num_frames)
        print(water.num_trajectory)
        print(water.num_frames_trajectory(0))
        # print(water.frames(1, [0]))

        f = water.frames(1, [0])
        p = f[0]['pc']
        b = f[0]['3d_bbox']
        p2, _ = p.points_in_box(b)
        print(p2.points.shape)


if __name__ == '__main__':
    unittest.main()
