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
    parser.add_argument('--cfg', type=str, default=r"D:\git_repo\PCMTT\config\PGNN.yaml", help='the config_file')
    parser.add_argument('--test', action='store_true', default=False, help='test mode')
    parser.add_argument('--preloading', action='store_true', default=False, help='preload dataset into memory')

    args = parser.parse_args(args=[])
    config = load_yaml(args.cfg)
    config.update(vars(args))
    return EasyDict(config)


class MyTestCase(unittest.TestCase):
    def test_kitti(self):

        cfg = parse_config()
        cfg.path = r"E:\dataset\kitti"
        kitti = KITTI_Util(cfg)
        print(kitti.num_frames)
        print(kitti.num_scenes)
        print(kitti.num_trajectory)
        # 初始化点云数
        kitti_point_num_inbox = 0
        kitti_point_num_inframe = 0
        kitti_point_inbox_counts = {}
        kitti_point_inframe_counts = {}
        for trajID in range(kitti.num_trajectory):
            for frameID in range(kitti.num_frames_trajectory(trajID)):
                # 如果frameID是从0开始的，需要加1
                adjusted_frameID = frameID + 1
                try:
                    # 获取当前轨迹和帧的数据处理结果
                    frame_data = kitti.frames(trajID, [frameID])
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
                            kitti_point_num_inbox = kitti_point_num_inbox + p2.points.shape[1]
                            kitti_point_num_inframe = kitti_point_num_inframe + pc1.points.shape[1]
                        # else:
                            print(f"Missing 'pc' or '3d_bbox' data for trajectory {trajID}, frame {frameID}.")
                            # 累加点云数
                            kitti_point_inbox_counts[(trajID, adjusted_frameID)] = p2.points.shape[1]
                            kitti_point_inframe_counts[(trajID, adjusted_frameID)] = pc1.points.shape[1]
                            # trajectory_point_counts[trajID] += p2.points.shape[1]
                            kitti_point_num_inbox += p2.points.shape[1]
                            kitti_point_num_inframe += pc1.points.shape[1]
                except IndexError as e:
                    # 处理索引错误
                    print(f"IndexError occurred: {e}")
            # 箱子内点的平均数量??? kitti.num_frames??箱子数量
        kitti_mean_point_inbox = kitti_point_num_inbox / kitti.num_frames
        # 帧图片点的平均数量
        kitti_mean_point_inframe = kitti_point_num_inframe / kitti.num_frames
        # 计算平均点云数
        kitti_mean_point_inbox = kitti_point_num_inbox / kitti.num_frames
        kitti_mean_point_inframe = kitti_point_num_inframe / kitti.num_frames
        # print(water_point_counts)
        print(kitti_point_num_inbox)
        print(kitti_mean_point_inbox)
        print(kitti_mean_point_inframe)
        print(kitti_point_inbox_counts)
        print(kitti_point_inframe_counts)
        kitti_frame_point_inbox_counts_sum = sum(kitti_point_inbox_counts.values())
        kitti_frame_point_inframe_counts_sum=sum(kitti_point_inframe_counts.values())
        # 打印总点云数
        print(f"The total sum of point inbox is: {kitti_frame_point_inbox_counts_sum}")
        print(f"The total sum of point inframe is: {kitti_frame_point_inframe_counts_sum}")
        return kitti_mean_point_inframe, kitti_mean_point_inbox, kitti_point_num_inbox, kitti_point_inbox_counts,kitti_point_inframe_counts
    #
    #
    def test_waterscene(self):

        cfg = parse_config()
        cfg.path = r"E:\dataset\waterScene"
        Water = WaterScene_Util(cfg)
        print(Water.num_frames)
        print(Water.num_scenes)
        print(Water.num_trajectory)
        # 初始化点云数
        water_point_num_inbox = 0
        water_point_num_inframe = 0
        water_frame_point_inbox_counts = {}
        water_frame_point_inframe_counts = {}
        for trajID in range(Water.num_trajectory):
            for frameID in range(Water.num_frames_trajectory(trajID)):
                # 如果frameID是从0开始的，需要加1
                adjusted_frameID = frameID + 1
                try:
                    # 获取当前轨迹和帧的数据处理结果
                    frame_data = Water.frames(trajID, [frameID])
                    # 检查frame_data是否是一个列表，且列表中至少有一个元素
                    frame_data = Water.frames(trajID, [frameID])
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
                            water_point_num_inbox = water_point_num_inbox + p2.points.shape[1]
                            water_point_num_inframe = water_point_num_inframe + pc1.points.shape[1]
                        # else:
                            print(f"Missing 'pc' or '3d_bbox' data for trajectory {trajID}, frame {frameID}.")
                            # 累加点云数
                            water_frame_point_inbox_counts[(trajID, adjusted_frameID)] = p2.points.shape[1]
                            water_frame_point_inframe_counts[(trajID, adjusted_frameID)] = pc1.points.shape[1]
                            # trajectory_point_counts[trajID] += p2.points.shape[1]
                            water_point_num_inbox += p2.points.shape[1]
                            water_point_num_inframe += pc1.points.shape[1]
                except IndexError as e:
                    # 处理索引错误
                    print(f"IndexError occurred: {e}")
            # 箱子内点的平均数量??? kitti.num_frames??箱子数量
        water_mean_point_inbox = water_point_num_inbox / Water.num_frames
        # 帧图片点的平均数量
        water_mean_point_inframe = water_point_num_inframe / Water.num_frames
        # 计算平均点云数
        water_mean_point_inbox = water_point_num_inbox / Water.num_frames
        water_mean_point_inframe = water_point_num_inframe / Water.num_frames
        # print(water_point_counts)
        print(water_point_num_inbox)
        print(water_mean_point_inbox)
        print(water_mean_point_inframe)
        print(water_frame_point_inbox_counts)
        print(water_frame_point_inframe_counts)
        water_frame_point_inbox_counts_sum = sum(water_frame_point_inbox_counts.values())
        water_frame_point_inframe_counts_sum=sum(water_frame_point_inframe_counts.values())
        # 打印总点云数
        print(f"The total sum of point inbox is: {water_frame_point_inbox_counts_sum}")
        print(f"The total sum of point inframe is: {water_frame_point_inframe_counts_sum}")
        return water_mean_point_inframe, water_mean_point_inbox, water_point_num_inbox, water_frame_point_inbox_counts,water_frame_point_inframe_counts


if __name__ == '__main__':
    unittest.main()
