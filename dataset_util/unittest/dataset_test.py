import unittest
from dataset_util.kitti import KITTI_Util
from dataset_util.waterscene import WaterScene_Util


class MyTestCase(unittest.TestCase):
    def test_kitti(self):
        kitti = KITTI_Util("H:/fyp/Open3DSOT/kitti", "train tiny", preloading=True)
        print(kitti.num_scenes)
        print(kitti.num_frames)
        print(kitti.num_trajecktory)
        print(kitti.num_frames_trajecktory(10))
        print(kitti.frames(0, [0]))

    def test_waterscene(self):
        ...
        # water = WaterScene_Util("")

if __name__ == '__main__':
    unittest.main()
