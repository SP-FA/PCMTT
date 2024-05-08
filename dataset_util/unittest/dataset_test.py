import yaml
import argparse
import unittest
from easydict import EasyDict
from dataset_util.kitti import KITTI_Util


def load_yaml(file_name):
    with open(file_name, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    return config


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default="../../config/PGNN_KITTI.yaml", help='the config_file')
    parser.add_argument('--test', action='store_true', default=False, help='test mode')
    parser.add_argument('--preloading', action='store_true', default=False, help='preload dataset into memory')

    args = parser.parse_args()
    config = load_yaml(args.cfg)
    config.update(vars(args))
    return EasyDict(config)


class MyTestCase(unittest.TestCase):
    def test_kitti(self):
        cfg = parse_config()
        cfg.path = "H:/E_extension/dataset/kitti"
        kitti = KITTI_Util(cfg, cfg.train_split)
        print(kitti.num_scenes)
        print(kitti.num_frames)
        print(kitti.num_trajectory)
        for i in range(kitti.num_trajectory):
            for j in range(kitti.num_frames_trajectory(i)):
                f = kitti.frames(i, [j])
                pc = f[0]['pc']
                print(f"{i} {j} {pc.points.shape}")


if __name__ == '__main__':
    unittest.main()
