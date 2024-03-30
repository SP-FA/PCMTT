import numpy as np
import pandas as pd
import os
import pickle
from collections import defaultdict
from pyquaternion import Quaternion
from dataset_util.base_class import BaseDataset
from dataset_util.point_struct import KITTI_PointCloud
from dataset_util.box_struct import Box


class KITTI_Util(BaseDataset):
    def __init__(self, path, split, **kwargs):
        super().__init__(path, split, **kwargs)
        self._KITTI_root = path
        self._coordinate_mode = "velodyne"
        self._KITTI_velo = os.path.join(path, "velodyne")
        # self._KITTI_img = os.path.join(path, "image_02")
        self._KITTI_label = os.path.join(path, "label_2")
        self._KITTI_calib = os.path.join(path, "calib")
        self._scene_list = self._get_scene_list(split)
        self._velos = defaultdict(dict)
        self._calibs = {}
        self._traj_list, self._traj_len_list = self._get_trajectory()
        if self._preloading:
            self._trainingSamples = self._load_data()

    @property
    def scenes_list(self):
        return self._scene_list

    @property
    def num_scenes(self):
        return len(self._scene_list)

    @property
    def num_trajectory(self):
        return len(self._traj_list)

    @property
    def num_frames(self):
        return sum(self._traj_len_list)

    def num_frames_trajectory(self, trajID):
        return self._traj_len_list[trajID]

    def frames(self, trajID, frameIDs):
        if self._preloading:
            frames = [self._trainingSamples[trajID][frameID] for frameID in frameIDs]
        else:
            traj = self._traj_list[trajID]
            frames = [self._get_frames_from_target(traj[frameID]) for frameID in frameIDs]
        return frames


    def _load_data(self):
        preloadPath = os.path.join(self._KITTI_root,
                                   f"preload_kitti_{self._split}_{self._coordinate_mode}_{self._preload_offset}.dat")
        if os.path.isfile(preloadPath):
            with open(preloadPath, 'rb') as f:
                trainingSamples = pickle.load(f)
        else:
            trainingSamples = []
            for i in range(len(self._traj_list)):
                frames = []
                for target in self._traj_list[i]:
                    frames.append(self._get_frames_from_target(target))
                trainingSamples.append(frames)
            with open(preloadPath, 'wb') as f:
                pickle.dump(trainingSamples, f)
        return trainingSamples

    def _get_trajectory(self):
        """获取所有目标的轨迹

        Returns:
            List[List[DataFrame]]: 一条 DataFrame 表示了一个目标在某一帧的信息。
                                   一个 List[DataFrame] 表示一个目标的轨迹。
                                   一个 traj_list 表示所有目标的轨迹信息。
            List[int]
        """
        traj_list = []
        traj_len_list = []
        for scene in self._scene_list:
            labelFile = os.path.join(self._KITTI_label, scene + ".txt")
            df = pd.read_csv(labelFile, sep=" ",
                             names=["frame", "track_id", "type", "truncated", "occluded",
                                    "alpha", "bbox_left", "bbox_top", "bbox_right", "bbox_bottom",
                                    "height", "width", "length", "x", "y", "z", "rotation_y"])
            df = df[df["type"] != 'DontCare']
            df.insert(loc=0, column="scene", value=scene)
            for trackID in df.track_id.unique():
                df_traj = df[df["track_id"] == trackID]
                df_traj = df_traj.sort_values(by=["frame"])
                df_traj = df_traj.reset_index(drop=True)
                trajectory = [traj for id, traj in df_traj.iterrows()]
                traj_list.append(trajectory)
                traj_len_list.append(len(trajectory))
        return traj_list, traj_len_list

    def _get_frames_from_target(self, target):
        """从某个目标的某一帧用获取点云和 box 信息 (frame)
           暂时不处理图像。
           根据 self.coordinate_mode 分成两种模式：
               1. "velodyne": 转换成空间坐标系下的 bbox
               2. "camera": 使用相机坐标系下的 bbox (原 label 中的 bbox 坐标是在相机坐标系下的)

        Args:
            target (DataFrame): 某一帧中的一个目标

        Returns:
            Dict {
                "pc": ,
                "3d_bbox": ,
                "meta": DataFrame
            }
        """
        sceneID = target["scene"]
        frameID = target["frame"]
        if sceneID in self._calibs.keys():
            calib = self._calibs[sceneID]
        else:
            calibPath = os.path.join(self._KITTI_calib, sceneID + ".txt")
            calib = self._read_calib(calibPath)
            self._calibs[sceneID] = calib

        velo_to_cam = np.vstack((calib["Tr_velo_cam"], np.array([0, 0, 0, 1])))
        if self._coordinate_mode == "velodrome":
            box_center_cam = np.array([target["x"], target["y"] - target["height"] / 2, target["z"], 1])
            box_center_velo = np.dot(np.linalg.inv(velo_to_cam), box_center_cam)
            box_center_velo = box_center_velo[:3]
            size = [target["width"], target["length"], target["height"]]
            orientation = Quaternion(
                axis=[0, 0, -1], radians=target["rotation_y"]) * Quaternion(axis=[0, 0, -1], degrees=90)
            bb = Box(box_center_velo, size, orientation)
        else:
            center = [target["x"], target["y"] - target["height"] / 2, target["z"]]
            size = [target["width"], target["length"], target["height"]]
            orientation = Quaternion(
                axis=[0, 1, 0], radians=target["rotation_y"]) * Quaternion(
                axis=[1, 0, 0], radians=np.pi / 2)
            bb = Box(center, size, orientation)

        try:
            if sceneID in self._velos.keys() and frameID in self._velos[sceneID].keys():
                pc = self._velos[sceneID][frameID]
            else:
                velodyne_path = os.path.join(self._KITTI_velo, sceneID, '{:06}.bin'.format(frameID))
                pc = KITTI_PointCloud(np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4).T)
                if self._coordinate_mode == "camera":
                    pc.transform(velo_to_cam)
                self._velos[sceneID][frameID] = pc
            # if self.preload_offset > 0:
            #     pc = points_utils.crop_pc_axis_aligned(pc, bb, offset=self.preload_offset)
        except:
            print(f"The point cloud at scene {sceneID} frame {frameID} is missing.")
            pc = KITTI_PointCloud(np.array([[0, 0, 0]]).T)
        return {"pc": pc, "3d_bbox": bb, "meta": target}

    @staticmethod
    def _get_scene_list(split):
        if "tiny" in split.lower():
            splitDict = {"train": [0], "valid": [18], "test": [19]}
        else:
            splitDict = {
                "train": list(range(0, 17)),
                "valid": list(range(17, 19)),
                "test": list(range(19, 21))}

        if "train" in split.lower():
            sceneNames = splitDict["train"]
        elif "valid" in split.lower():
            sceneNames = splitDict["valid"]
        elif "test" in split.lower():
            sceneNames = splitDict["test"]
        else:
            sceneNames = list(range(21))

        sceneNames = ["%04d" % sceneName for sceneName in sceneNames]
        return sceneNames

    @staticmethod
    def _read_calib(path):
        data = {}
        with open(path, 'r') as f:
            for line in f.readlines():
                values = line.split()
                try:
                    data[values[0]] = np.array([float(x) for x in values[1:]]).reshape(3, 4)
                except ValueError:
                    pass
        return data
