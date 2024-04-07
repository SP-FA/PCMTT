import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dataset_util.unittest.dataset_test import MyTestCase

test_case = MyTestCase()
mean_point_inframe, mean_point_inbox, point_counts, num_frames, point_num_inbox,frame_point_cloud_counts = test_case.test_kitti()

dic = frame_point_cloud_counts
dic_keys = list(dic.keys())
dic_values= list(dic.values())
print(len(dic_keys))

i = 1
for key in dic_keys:
    new_key = f"{i}"
    dic[new_key] = dic[key]
    del dic[key]
    i=i+1

# 使用字典推导式创建一个新字典，只包含值大于10的键值对
new_dic = {k: v for k, v in dic.items() if v > 10}
print(f"dic{new_dic.values()}")
dic_keys_new=list(new_dic.keys())
df = pd.DataFrame({
    "frame_num":dic_keys_new,
    'Point Count': list(new_dic.values())
})

plt.figure(figsize=(10, 6))
sns.barplot(x="frame_num", y='Point Count',hue="frame_num",palette="viridis",data=df, alpha=0.8)
# Add title and labels
plt.title('Point Cloud Count Distribution In Frames')
plt.xlabel('Point Cloud Count')
plt.ylabel('Frame Number')
plt.tight_layout()
print(f"最少点云数{min(list(new_dic.values ()))}")
print(f"最多点云数{max(list(new_dic.values ()))}")
# Show the plot
plt.show()