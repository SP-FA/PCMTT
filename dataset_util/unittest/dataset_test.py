import unittest

from dataset_util.kitti import KITTI_Util
from dataset_util.waterscene import WaterScene_Util

class MyTestCase(unittest.TestCase):

    def test_kitti(self):
        kimmi = KITTI_Util(r"C:\Users\49465\Desktop\dataset_mini\kitti_mini", "train tiny", preloading=True)
        print("kitti.num_scenes:{}".format(kimmi.num_scenes))
        print("kitti.num_frames:{}".format(kimmi.num_frames))
        print("kitti.num_trajectory:{}".format(kimmi.num_trajectory))
        print("kitti.num_frames_trajectory:{}".format(kimmi.num_frames_trajectory(10)))
        point_num_inbox = 0
        point_num_inframe = 0
        for trajID in range(kimmi.num_trajectory):
            for frameID in range(kimmi.num_frames_trajectory(trajID)):
                try:
                    # 获取当前轨迹和帧的数据处理结果
                    frame_data = kimmi.frames(trajID, [frameID])
                    # 检查frame_data是否是一个列表，且列表中至少有一个元素
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
                except IndexError as e:
                    # 处理索引错误
                    print(f"IndexError occurred: {e}")
        # 箱子内点的平均数量??? kitti.num_frames??箱子数量
        mean_point_inbox = point_num_inbox / kimmi.num_frames
        # 帧图片点的平均数量
        mean_point_inframe = point_num_inframe / kimmi.num_frames
        print(mean_point_inbox)
        print(mean_point_inframe)

        return mean_point_inframe, mean_point_inbox, point_num_inbox, point_num_inframe
    #
    # def test_waterscene(self):
    #     water = WaterScene_Util("H:/E_extension/dataset/waterScene", "train tiny", preloading=False)
    #     print("kitti.num_scenes:{}".format(water.num_scenes))
    #     print("kitti.num_frames:{}".format(water.num_frames))
    #     print("kitti.num_trajectory:{}".format(water.num_trajectory))
    #     print("kitti.num_frames_trajectory:{}".format(water.num_frames_trajectory(10)))
    #     point_num_inbox = 0
    #     point_num_inframe = 0
    #     for trajID in range(water.num_trajectory):
    #         for frameID in range(water.num_frames_trajectory(trajID)):
    #             try:
    #                 # 获取当前轨迹和帧的数据处理结果
    #                 frame_data = water.frames(trajID, [frameID])
    #                 # 检查frame_data是否是一个列表，且列表中至少有一个元素
    #                 if isinstance(frame_data, list) and len(frame_data) > 0:
    #                     # 尝试访问列表中的第一个元素，并获取'pc'键对应的点云数据
    #                     pc1 = frame_data[0].get('pc')
    #                     # 获取'3d_bbox'键对应的3D边界框数据
    #                     bbox1 = frame_data[0].get('3d_bbox')
    #                     # 确保成功获取了点云和边界框
    #                     if pc1 is not None and bbox1 is not None:
    #                         # 调用点云数据的points_in_box方法，计算在边界框内的点云
    #                         p2, _ = pc1.points_in_box(bbox1)
    #                         # 打印相关信息
    #                         print("Frame data:", frame_data)
    #                         print("Point cloud shape:", pc1.points.shape)
    #                         print("Points in box shape:", p2.points.shape[1])
    #                         point_num_inbox = point_num_inbox + p2.points.shape[1]
    #                         point_num_inframe = point_num_inframe + pc1.points.shape[1]
    #                     else:
    #                         print(f"Missing 'pc' or '3d_bbox' data for trajectory {trajID}, frame {frameID}.")
    #             except IndexError as e:
    #                 # 处理索引错误
    #                 print(f"IndexError occurred: {e}")
    #     # 箱子内点的平均数量??? kitti.num_frames??箱子数量
    #     mean_point_inbox = point_num_inbox / water.num_frames
    #     # 帧图片点的平均数量
    #     mean_point_inframe = point_num_inframe / water.num_frames
    #     print(mean_point_inbox)
    #     print(mean_point_inframe)
    #     return mean_point_inframe, mean_point_inbox, point_num_inbox, point_num_inframe


if __name__ == '__main__':
    unittest.main()
