import yaml
import argparse
from easydict import EasyDict


def load_yaml(file_name):
    with open(file_name, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    return config


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default="./config/PGNN.yaml", help='the config_file')
    parser.add_argument('--test', action='store_true', default=False, help='test mode')
    parser.add_argument('--preloading', action='store_true', default=False, help='preload dataset into memory')

    args = parser.parse_args()
    config = load_yaml(args.cfg)
    config.update(vars(args))
    return EasyDict(config)


if __name__ == "__main__":
    cfg = parse_config()
    print(cfg)
