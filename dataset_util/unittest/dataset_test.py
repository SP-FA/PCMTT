import numpy as np
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
    parser.add_argument('--cfg', type=str, default="../../config/PGNN.yaml", help='the config_file')
    parser.add_argument('--test', action='store_true', default=False, help='test mode')
    parser.add_argument('--preloading', action='store_true', default=False, help='preload dataset into memory')

    args = parser.parse_args()
    config = load_yaml(args.cfg)
    config.update(vars(args))
    return EasyDict(config)


class MyTestCase(unittest.TestCase):
    def test_kitti(self):
        cfg = parse_config()
        cfg.path = r"E:\dataset\kitti"
        kitti = KITTI_Util(cfg)
        print(kitti.num_scenes)
        print(kitti.num_frames)
        print(kitti.num_trajectory)
        print(kitti.num_frames_trajectory(10))
        print(kitti.frames(0, [0]))
        # 初始化点云数
        point_num_inbox = 0
        point_num_inframe = 0
        # 定义帧数范围
        num_frames = self.kitti.num_frames
        print(num_frames)
        percentile_bins = np.linspace(0, 100, 9)
        frame_ranges = {}
        for i in range(len(percentile_bins) - 1):
            start_idx = np.searchsorted(np.arange(1, num_frames + 1), percentile_bins[i])
            end_idx = np.searchsorted(np.arange(1, num_frames + 1), percentile_bins[i + 1]) - 1
            if i == len(percentile_bins) - 2:
                end_idx = num_frames
            frame_ranges[
                f"{int(percentile_bins[i]) * num_frames / 100} - {int(percentile_bins[i + 1]) * num_frames / 100}"] = (
                start_idx, end_idx)

        # 初始化每个范围内的点云数为0
        point_counts = {range_label: 0 for range_label in frame_ranges}
        frame_point_cloud_counts = {}
        for trajID in range(self.kitti.num_trajectory):
            for frameID in range(self.kitti.num_frames_trajectory(trajID)):
                # 如果frameID是从0开始的，需要加1
                adjusted_frameID = frameID + 1
                try:
                    # 获取当前轨迹和帧的数据处理结果
                    frame_data = kitti.frames(trajID, [frameID])
                    # 检查frame_data是否是一个列表，且列表中至少有一个元素
                    frame_data = self.kitti.frames(trajID, [frameID])
                    if isinstance(frame_data, list) and len(frame_data) > 0:
                        # 尝试访问列表中的第一个元素，并获取'pc'键对应的点云数据
                        pc1 = frame_data[0].get('pc')
                        # 获取'3d_bbox'键对应的3D边界框数据
                        bbox1 = frame_data[0].get('3d_bbox')
                        # 确保成功获取了点云和边界框
                        if pc1 is not None and bbox1 is not None:
                            # 调用点云数据的points_in_box方法，计算在边界框内的点云
                            p2, _ = pc1.points_in_box(bbox1)
                            # 打印相关信息
                            print("Frame data:", frame_data)
                            print("Point cloud shape:", pc1.points.shape)
                            print("Points in box shape:", p2.points.shape[1])
                            point_num_inbox = point_num_inbox + p2.points.shape[1]
                            point_num_inframe = point_num_inframe + pc1.points.shape[1]
                        else:
                            print(f"Missing 'pc' or '3d_bbox' data for trajectory {trajID}, frame {frameID}.")
                            # 累加点云数
                            frame_point_cloud_counts[(trajID, adjusted_frameID)] = p2.points.shape[1]
                            # trajectory_point_counts[trajID] += p2.points.shape[1]
                            point_num_inbox += p2.points.shape[1]
                            point_num_inframe += pc1.points.shape[1]
                            print("Start")
                            for range_label, (start_idx, end_idx) in frame_ranges.items():
                                print(f"Current range: {range_label}, Start: {start_idx}, End: {end_idx}")
                                if start_idx <= adjusted_frameID <= end_idx:
                                    point_counts[range_label] += p2.points.shape[1]
                                    break
                            print("Finished")
                # except Exception as e:
                #     print(f"Exception occurred for trajectory {trajID}, frame {frameID}: {e}")
                except IndexError as e:
                    # 处理索引错误
                    print(f"IndexError occurred: {e}")
            # 箱子内点的平均数量??? kitti.num_frames??箱子数量
        mean_point_inbox = point_num_inbox / kitti.num_frames
        # 帧图片点的平均数量
        mean_point_inframe = point_num_inframe / kitti.num_frames
        # 计算平均点云数
        mean_point_inbox = point_num_inbox / self.kitti.num_frames
        mean_point_inframe = point_num_inframe / self.kitti.num_frames
        # # 打印每个范围内的点云数
        # for range_label, count in point_counts.items():
        #     print(f"Point cloud count for range {range_label}: {count}")

        print(point_counts)
        print(point_num_inbox)
        print(mean_point_inbox)
        print(mean_point_inframe)
        print(frame_point_cloud_counts)
        total_point_count = sum(frame_point_cloud_counts.values())
        # 打印总点云数
        print(f"The total sum of point clouds is: {total_point_count}")
        return mean_point_inframe, mean_point_inbox, point_counts, num_frames, point_num_inbox, frame_point_cloud_counts
    #
    # def test_waterscene(self):
    #     cfg = parse_config()
    #     cfg.preloading = False
    #     water = WaterScene_Util(cfg)
    #     print(water.num_scenes)
    #     print(water.num_frames)
    #     print(water.num_trajectory)
    #     print(water.num_frames_trajectory(0))
    #     # print(water.frames(1, [0]))
    #
    #     f = water.frames(1, [0])
    #     p = f[0]['pc']
    #     b = f[0]['3d_bbox']
    #     p2, _ = p.points_in_box(b)
    #     print(p2.points.shape)


if __name__ == '__main__':
    unittest.main()
