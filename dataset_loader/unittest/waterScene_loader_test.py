import yaml
import unittest
import argparse
from easydict import EasyDict
from dataset_loader.waterScene_loader import WaterScene_Loader
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
    def test_waterScene_loader(self):
        cfg = parse_config()
        cfg.path = "H:/E_extension/dataset/waterScene"
        cfg.preloading = True
        water = WaterScene_Util(cfg)
        print(water.num_scenes)
        print(water.num_frames)
        print(water.num_trajectory)
        print(water.num_frames_trajectory(0))

        wl = WaterScene_Loader(water, cfg)
        for i in wl:
            print(i)
            break


if __name__ == '__main__':
    unittest.main()
