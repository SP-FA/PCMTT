# Point Cloud Muti-Target Tracking

毕业设计，水面点云多目标追踪

项目完成之前，该 reademe 暂时用作开发文档

### dataset_util (SPFA)

- base_class: 定义了Bound-Box、点云数据结构和数据集 util 的基类
    - BaseDataset:
        - num_scenes: 有几段视频
        - num_trajecktory: 轨迹的数量
        - num_frames: 所有轨迹的总帧数
        - num_frames_trajecktory(self, trajID: int) -> int: 某个轨迹的帧数
        - frames(self, trajID: int, frameIDs: int) -> List[Dict["pc": pc, "3d_bbox": bb, "meta": target]]: 获取某个轨迹的某几帧
    - BasePointCloud
        - n: 点的维度
        - subsample(self, ratio: float): 下采样
        - points_in_box(self, box: Box, returnMask=False) -> WaterScene_PointCloud / KITTI_PointCloud, Optional[Tensor]: 返回在 box 内的点和 mask
        - convert2Tensor(self) -> torch.tensor: 类型转换
        - fromTensor(cls, tensor: torch.tensor) -> BasePointCloud: 类型转换
- kitti: 定义了 KITTI 数据集的 util 方法
    - KITTI_Util
        - __init__(self, path, split, **kwargs)
- waterscene: 定义了 waterscene 数据集的 util 方法
    - WaterScene_Util
        - __init__(self, path, split, **kwargs)
- data_struct: 定义了 Box 和 PointCloud 的数据结构
    - WaterScene_PointCloud: 定义了 waterscene 数据集的点云结构
        - __init__(self, points)
        - from_file(cls, fileName) -> WaterScene_PointCloud: 通过文件名获取点云
    - KITTI_PointCloud: 定义了 KITTI 数据集的点云结构
        - __init__(self, points)
        - from_file(cls, fileName) -> KITTI_PointCloud
        - translate(self, x: List[float * 3])
        - rotate(self, rot_mat)
        - transform(self, trans_mat)
        - normalize(self, wlh: List[float * 3])
    - Box: 定义了 Bound-Box 的数据结构
        - __init__(self, center, size, orient, label=np.nan, score=np.nan,
                    veloc=(np.nan, np.nan, np.nan))
        - __eq__(self, other)
        - __repr__(self)
        - encode(self) -> List[float * 16]
        - decode(cls, data: List[float * 16]) -> Box
        - rotation_matrix: 获取旋转矩阵
        - corners(self, wlh_factor: float=1.0) -> np.array[3, 8]
        - bottom_corners(self) -> np.array[3, 4]
        - translate(self, x: Tuple[float * 3])
        - rotate(self, quaternion: Quaternion)
        - transform(self, trans_mat)

### dataset_loader (dehudewf)

