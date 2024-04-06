import os
import json
import pickle
from collections import defaultdict

import pandas as pd
import numpy as np

from pyquaternion import Quaternion
from dataset_util.base_class import BaseDataset
from dataset_util.point_struct import WaterScene_PointCloud, KITTI_PointCloud
from dataset_util.box_struct import Box


class WaterScene_Util(BaseDataset):
    def __init__(self, path, split, **kwargs):
        super().__init__(path, split, **kwargs)
        self._WaterScene_root = path
        self._coordinate_mode = "velodyne"
        self._scene_list = self._get_scene_list(split)
        self._velos = defaultdict(dict)
        self._calibs = {}
        self._traj_list, self._traj_len_list = self._get_trajectory()
        if self._preloading:
            self._trainingSamples = self._load_data()


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
            frames = [self._get_frame_from_target(traj[frameID]) for frameID in frameIDs]
        return frames

    def _load_data(self):
        preloadPath = os.path.join(self._WaterScene_root, f"preload_waterscene_{self._split}_{self._coordinate_mode}_{self._preload_offset}.dat")
        if os.path.isfile(preloadPath):
            with open(preloadPath, 'rb') as f:
                trainingSamples = pickle.load(f)
        else:
            trainingSamples = []
            for i in range(len(self._traj_list)):
                frames = []
                for target in self._traj_list[i]:
                    frames.append(self._get_frame_from_target(target))
                trainingSamples.append(frames)
            with open(preloadPath, 'wb') as f:
                pickle.dump(trainingSamples, f)
        return trainingSamples

    def _get_trajectory(self):
        traj_list = []
        traj_len_list = []
        for scene in self._scene_list:
            labelPath = os.path.join(self._WaterScene_root, scene, "label")
            labelList = []
            for labelFile in os.listdir(labelPath):
                with open(os.path.join(labelPath, labelFile), 'r') as f:
                    content = f.read()
                    jss = json.loads(content)
                    for js in jss:
                        labelDict = {
                            "frame": labelFile[:-5],
                            "track_id": js["obj_id"],
                            "type": "Unknown",
                            "truncated": 0,
                            "occluded": 0,
                            "alpha": 0,
                            "bbox_left": 0,
                            "bbox_top": 0,
                            "bbox_right": 0,
                            "bbox_bottom": 0,
                            "height": js["psr"]["scale"]["z"],  # z
                            "width": js["psr"]["scale"]["x"],  # x
                            "length": js["psr"]["scale"]["y"],  # y
                            "x": js["psr"]["position"]["x"],
                            "y": js["psr"]["position"]["y"],
                            "z": js["psr"]["position"]["z"],
                            "rotation_y": js["psr"]["rotation"]["z"]  # rotation_y 指的是在相机坐标系下的 y 轴
                        }
                        labelDF = pd.DataFrame([labelDict])
                        labelList.append(labelDF)
            df = pd.concat(labelList)
            df.insert(loc=0, column="scene", value=scene)
            for trackID in df.track_id.unique():
                df_traj = df[df["track_id"] == trackID]
                df_traj = df_traj.sort_values(by=["frame"])
                df_traj = df_traj.reset_index(drop=True)
                trajectory = [traj for id, traj in df_traj.iterrows()]
                traj_list.append(trajectory)
                traj_len_list.append(len(trajectory))
        return traj_list, traj_len_list

    def _get_frame_from_target(self, target):
        sceneID = target["scene"]
        frameID = target["frame"]
        if sceneID in self._calibs.keys():
            calib = self._calibs[sceneID]
        else:
            calibPath = os.path.join(self._WaterScene_root, sceneID, "calib", "calib.txt")
            calib = self._read_calib(calibPath)
            self._calibs[sceneID] = calib

        velo2Cam = np.vstack((calib["t_camera_intrinsic"], np.array([0, 0, 0, 1])))
        if self._coordinate_mode == "velodyne":
            # center = np.array([target["x"], target["y"] - target["height"] / 2, target["z"], 1])
            center = [target["x"], target["y"], target["z"]]
            # box_center_velo = np.dot(np.linalg.inv(velo2Cam), center)
            # box_center_velo = box_center_velo[:3]
            size = [target["width"], target["length"], target["height"]]
            # orientation = Quaternion(
            #     axis=[0, 0, -1], radians=target["rotation_y"]) * Quaternion(
            #         axis=[0, 0, -1], degrees=90)
            orientation = Quaternion(axis=[0, 0, 1], radians=target["rotation_y"])
            bb = Box(center, size, orientation)
        else:
            ...

        try:
            if sceneID in self._velos.keys() and frameID in self._velos[sceneID].keys():
                pc = self._velos[sceneID][frameID]
            else:
                velodynePath = os.path.join(self._WaterScene_root, sceneID, "radar", f"{frameID}.csv")
                pc = WaterScene_PointCloud.from_file(velodynePath)
                # if self.coordinate_mode == "camera":
                #     pc.transform(velo_2_cam)
                self._velos[sceneID][frameID] = pc
            # if self.preload_offset > 0:
            #     pc = points_utils.crop_pc_axis_aligned(pc, bb, offset=self.preload_offset)
        except:
            print(f"The point cloud at scene {sceneID} frame {frameID} is missing.")
            pc = WaterScene_PointCloud(np.zeros(8).reshape(-1, 8))
        return {"pc": pc, "3d_bbox": bb, "meta": target}

    @staticmethod
    def _get_scene_list(split):
        if "tiny" in split.lower():
            splitDict = {"train": [2], "valid": [8], "test": [17]}
        else:
            splitDict = {"train": [2, 6, 7], "valid": [8], "test": [17]}
        
        if "train" in split.lower():
            sceneNames = splitDict["train"]
        elif "valid" in split.lower():
            sceneNames = splitDict["valid"]
        elif "test" in split.lower():
            sceneNames = splitDict["test"]

        sceneNames = [str(name) for name in sceneNames]
        return sceneNames

    @staticmethod
    def _read_calib(path):
        data = {}
        with open(path, 'r') as f:
            for line in f.readlines():
                values = line.split()
                try:
                    data[values[0][:-1]] = np.array([float(x) for x in values[1:]]).reshape(3, 4)
                except ValueError:
                    data[values[0][:-1]] = np.array([float(x) for x in values[1:]]).reshape(4, 4)
                    # pass
            return data
