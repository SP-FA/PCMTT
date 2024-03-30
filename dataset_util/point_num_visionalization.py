import matplotlib.pyplot as plt
import seaborn as sns
from dataset_util.unittest.dataset_test import MyTestCase

mean_point_inframe, mean_point_inbox, point_num_inbox, point_num_inframe=MyTestCase.test_kitti()

plt.figure(figsize=(10, 10))
plt.subplot(2,2,1)

plt.title()
plt.grid(True)


plt.subplot(2,2,2)
plt.title()
plt.grid(True)


plt.subplot(2,2,3)
plt.title()
plt.grid(True)


plt.subplot(2,2,4)
plt.title()
plt.grid(True)
plt.show()