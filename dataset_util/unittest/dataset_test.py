import unittest
import numpy as np
from dataset_util.kitti import KITTI_Util
from dataset_util.waterscene import WaterScene_Util
class MyTestCase(unittest.TestCase):

    def test_kitti(self):
        # 初始化KITTI_Util对象
        self.kitti = KITTI_Util(r"E:\dataset\kitti", "train tiny", preloading=True)
        print("kitti.num_scenes:{}".format(self.kitti.num_scenes))
        print("kitti.num_frames:{}".format(self.kitti.num_frames))
        print("kitti.num_trajectory:{}".format(self.kitti.num_trajectory))
        print("kitti.num_frames_trajectory:{}".format(self.kitti.num_frames_trajectory(10)))
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
        # trajectory_point_counts = {trajID: 0 for trajID in range(self.kimmi.num_trajectory)}
        frame_point_cloud_counts = {}

        for trajID in range(self.kitti.num_trajectory):
            for frameID in range(self.kitti.num_frames_trajectory(trajID)):
                # 如果frameID是从0开始的，需要加1
                adjusted_frameID = frameID + 1
                try:
                    # 获取当前轨迹和帧的数据处理结果
                    frame_data = self.kitti.frames(trajID, [frameID])
                    if isinstance(frame_data, list) and len(frame_data) > 0:
                        pc1 = frame_data[0].get('pc')
                        bbox1 = frame_data[0].get('3d_bbox')
                        if pc1 is not None and bbox1 is not None:
                            p2, _ = pc1.points_in_box(bbox1)
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
                    print(f"IndexError occurred: {e}")
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

    # def test_kitti(self):
    #     # 初始化KITTI_Util对象
    #     self.Water = WaterScene_Util(r"E:\dataset\waterScene", "train tiny", preloading=True)
    #     print("kitti.num_scenes:{}".format(self.Water.num_scenes))
    #     print("kitti.num_frames:{}".format(self.Water.num_frames))
    #     print("kitti.num_trajectory:{}".format(self.Water.num_trajectory))
    #     print("kitti.num_frames_trajectory:{}".format(self.Water.num_frames_trajectory(10)))
    #
    #     # 初始化点云数
    #     Water_point_num_inbox = 0
    #     Water_point_num_inframe = 0
    #
    #     # 定义帧数范围
    #     Water_num_frames = self.Water.num_frames
    #     print(Water_num_frames)
    #     percentile_bins = np.linspace(0, 100, 9)
    #     frame_ranges = {}
    #     for i in range(len(percentile_bins) - 1):
    #         start_idx = np.searchsorted(np.arange(1, Water_num_frames + 1), percentile_bins[i])
    #         end_idx = np.searchsorted(np.arange(1, Water_num_frames + 1), percentile_bins[i + 1]) - 1
    #         if i == len(percentile_bins) - 2:
    #             end_idx = Water_num_frames
    #         frame_ranges[
    #             f"{int(percentile_bins[i]) * Water_num_frames / 100} - {int(percentile_bins[i + 1]) * Water_num_frames / 100}"] = (
    #             start_idx, end_idx)
    #
    #     # 初始化每个范围内的点云数为0
    #     Water_point_counts = {range_label: 0 for range_label in frame_ranges}
    #     # trajectory_point_counts = {trajID: 0 for trajID in range(self.kimmi.num_trajectory)}
    #     Water_frame_point_cloud_counts = {}
    #
    #     for trajID in range(self.Water.num_trajectory):
    #         for frameID in range(self.Water.num_frames_trajectory(trajID)):
    #             # 如果frameID是从0开始的，需要加1
    #             adjusted_frameID = frameID + 1
    #             try:
    #                 # 获取当前轨迹和帧的数据处理结果
    #                 frame_data = self.Water.frames(trajID, [frameID])
    #                 if isinstance(frame_data, list) and len(frame_data) > 0:
    #                     pc1 = frame_data[0].get('pc')
    #                     bbox1 = frame_data[0].get('3d_bbox')
    #                     if pc1 is not None and bbox1 is not None:
    #                         p2, _ = pc1.points_in_box(bbox1)
    #                         # 累加点云数
    #                         Water_frame_point_cloud_counts[(trajID, adjusted_frameID)] = p2.points.shape[1]
    #                         # trajectory_point_counts[trajID] += p2.points.shape[1]
    #                         Water_point_num_inbox += p2.points.shape[1]
    #                         Water_point_num_inframe += pc1.points.shape[1]
    #                         print("Start")
    #                         for range_label, (start_idx, end_idx) in frame_ranges.items():
    #                             print(f"Current range: {range_label}, Start: {start_idx}, End: {end_idx}")
    #                             if start_idx <= adjusted_frameID <= end_idx:
    #                                 Water_point_counts[range_label] += p2.points.shape[1]
    #                                 break
    #                         print("Finished")
    #             # except Exception as e:
    #             #     print(f"Exception occurred for trajectory {trajID}, frame {frameID}: {e}")
    #             except IndexError as e:
    #                 print(f"IndexError occurred: {e}")
    #     # 计算平均点云数
    #     Water_mean_point_inbox = Water_point_num_inbox / self.Water.num_frames
    #     Water_mean_point_inframe = Water_point_num_inframe / self.Water.num_frames
    #     # # 打印每个范围内的点云数
    #     # for range_label, count in point_counts.items():
    #     #     print(f"Point cloud count for range {range_label}: {count}")
    #
    #     print(Water_point_counts)
    #     print(Water_point_num_inbox)
    #     print(Water_mean_point_inbox)
    #     print(Water_mean_point_inframe)
    #     print(Water_frame_point_cloud_counts)
    #     total_point_count = sum(Water_frame_point_cloud_counts.values())
    #     # 打印总点云数
    #     print(f"The total sum of point clouds is: {total_point_count}")
    #     return Water_mean_point_inframe, Water_mean_point_inbox, Water_point_counts, Water_num_frames, Water_point_num_inbox, Water_frame_point_cloud_counts

if __name__ == '__main__':
    unittest.main()
