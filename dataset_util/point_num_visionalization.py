import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dataset_util.unittest.dataset_test import MyTestCase

test_case = MyTestCase()
water_mean_point_inframe, water_mean_point_inbox, water_point_num_inbox, water_point_inbox_counts, water_point_inframe_counts = test_case.test_waterscene()
kitti_mean_point_inframe, kitti_mean_point_inbox, kitti_point_num_inbox, kitti_point_inbox_counts, kitti_point_inframe_counts = test_case.test_kitti()


class point_num_vision:

    def __init__(self, dic, dic_key, dic_values):
        """
        argument:
        1.dic:water_frame_point_inbox_counts/water_frame_point_inframe_counts
        2.dic_key:list(dic.keys())
        3.dic_values:list(dic.values())
        """
        self.dic = dic
        self.dic_key = dic_key
        self.dic_values = dic_values

    def select_conditional_dic(self):
        i = 1
        for key in self.dic_key:
            new_key = f"{i}"
            self.dic[new_key] = self.dic[key]
            del self.dic[key]
            i = i + 1
        # 使用字典推导式创建一个新字典，只包含值大于10的键值对
        new_dic = {k: v for k, v in self.dic.items() if v > 10 & v < 10000}
        print(f"dic{new_dic.values()}")
        dic_keys_new = list(new_dic.keys())
        return new_dic

    def num_frame_in_point_interval(self, cloud_range):
        neww_dic = self.select_conditional_dic()
        # water_inframe
        # cloud_intervals = [(i * cloud_range+30, i * cloud_range + cloud_range+30) for i in
        #                    range(1, (max(neww_dic.values()) // cloud_range + 1)-1)]
        # kitti-inbox
        # cloud_intervals = [(i * cloud_range, i * cloud_range + cloud_range) for i in
        #                    range(0, (900 // cloud_range ))]
        # kitti_inframe
        cloud_intervals = [(i * cloud_range, i * cloud_range + cloud_range) for i in
                           range(33, (130000 // cloud_range))]
        # 创建一个字典来存储每个区间的帧数
        frame_counts_by_interval = {f"{start}-{end}": 0 for start, end in cloud_intervals}
        # 遍历字典，统计每个区间的帧数
        for frame, cloud_count in neww_dic.items():
            for start, end in cloud_intervals:
                if start <= cloud_count < end:
                    frame_counts_by_interval[f"{start}-{end}"] += 1
                    break

        df_inbox = pd.DataFrame({
            "interval": list(frame_counts_by_interval.keys()),
            "count": list(frame_counts_by_interval.values())
        })

        name = ""
        if self.dic == water_point_inframe_counts:
            name = "water_frame"
        elif self.dic == water_point_inbox_counts:
            name = "water_box"
        elif self.dic == kitti_point_inframe_counts:
            name = "kitti_frame"
        else:
            name = "kitti_box"

        sns.barplot(x="interval", y="count", hue="interval", palette="viridis", data=df_inbox, alpha=0.8)
        plt.title(f'Point Cloud in {name} Distribution')
        plt.xlabel(f'Point Cloud in {name}')
        plt.ylabel('Frame Number')
        plt.tight_layout()
        plt.show()


pv = point_num_vision(kitti_point_inframe_counts, list(kitti_point_inframe_counts.keys()),
                      list(kitti_point_inframe_counts.keys()))
pv.num_frame_in_point_interval(cloud_range=300)

# print(f"最少点云数{min(list(new_dic.values()))}")
# print(f"最多点云数{max(list(new_dic.values()))}")
#
# df_inbox = pd.DataFrame({
#     "frame_num_inbox": dic_keys_new,
#     'Point inbox': list(new_dic.values()),
# })
# df_inframe = pd.DataFrame({
#     "frame_num_inframe": dic_keys_new1,
#     "point inframe": list(new_dic1.values())
# })
#
# plt.figure(figsize=(10, 6))
# plt.subplot(2, 1, 1)
# sns.barplot(x="Point inbox", y='frame_num_inbox', hue="Point inbox", palette="viridis", data=df_inbox, alpha=0.8)
#
# # Add title and labels
# plt.title('Point Cloud in box Distribution In Frames')
# plt.xlabel('Point Cloud in box')
# plt.ylabel('Frame Number')
# plt.tight_layout()
# print(f"最少点云数{min(list(new_dic.values()))}")
# print(f"最多点云数{max(list(new_dic.values()))}")
#
# plt.subplot(2, 1, 2)
# sns.barplot(x="point inframe", y='frame_num_inframe', hue="point inframe", palette="viridis", data=df_inframe,
#             alpha=0.8)
# # ax.xaxis.set_major_locator(x_major_locator)  # 应用x轴的主刻度间隔
# plt.title('Point Cloud in frame Distribution In Frames')
# plt.xlabel('Point Cloud in frame')
# plt.ylabel('Frame Number')
# plt.tight_layout()
# print(f"最少点云数{min(list(new_dic1.values()))}")
# print(f"最多点云数{max(list(new_dic1.values()))}")
# # # Show the plot
# plt.show()
