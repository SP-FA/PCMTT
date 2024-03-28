import unittest

from dataset_loader.waterScene_loader import WaterScene_Loader
from dataset_util.waterscene import WaterScene_Util


class MyTestCase(unittest.TestCase):
    def test_waterScene_loader(self):
        water = WaterScene_Util("H:/E_extension/dataset/waterScene", "train tiny", preloading=False)
        print(water.num_scenes)
        print(water.num_frames)
        print(water.num_trajectory)
        print(water.num_frames_trajectory(0))

        wl = WaterScene_Loader(water)
        for i in wl:
            print(i)
            break


if __name__ == '__main__':
    unittest.main()
