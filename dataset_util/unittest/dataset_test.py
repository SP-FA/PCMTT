import yaml
import argparse
import unittest
from easydict import EasyDict
from dataset_util.kitti import KITTI_Util
from dataset_util.waterscene import WaterScene_Util


def load_yaml(file_name):
    with open(file_name, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    return config


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default="../../config/PGNN_WaterScene.yaml", help='the config_file')
    parser.add_argument('--test', action='store_true', default=False, help='test mode')
    parser.add_argument('--preloading', action='store_true', default=False, help='preload dataset into memory')

    args = parser.parse_args()
    config = load_yaml(args.cfg)
    config.update(vars(args))
    return EasyDict(config)


class MyTestCase(unittest.TestCase):
    def test_kitti(self):
        cfg = parse_config()
        cfg.path = "H:/fyp/Open3DSOT/kitti"
        kitti = KITTI_Util(cfg)
        print(kitti.num_scenes)
        print(kitti.num_frames)
        print(kitti.num_trajectory)
        print(kitti.num_frames_trajectory(10))
        print(kitti.frames(0, [0]))

    def test_waterscene(self):
        cfg = parse_config()
        cfg.preloading = False
        water = WaterScene_Util(cfg)
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
