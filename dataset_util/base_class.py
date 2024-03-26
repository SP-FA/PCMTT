import copy

import numpy as np
import torch
from pyquaternion import Quaternion

from dataset_util.data_struct import Box


class BaseDataset:
	def __init__(self, path, split, **kwargs):
		self.path = path
		self._split = split
		self._coordinate_mode = kwargs.get("coordinate_mode", "velogyne")
		self._preload_offset = kwargs.get("preload_offset", -1)
		self._preloading = kwargs.get('preloading', False)

	@property
	def num_scenes(self):
		raise NotImplementedError

	@property
	def num_trajecktory(self):
		raise NotImplementedError

	@property
	def num_frames(self):
		raise NotImplementedError

	def num_frames_trajecktory(self, trajID):
		raise NotImplementedError

	def frames(self, trajID, frameIDs):
		raise NotImplementedError


class BasePointCloud:
	def __init__(self, points, dim):
		"""
		Args:
			points (np.array[dim, n]) # dim 是点云维度
		"""
		if dim != points.shape[0]:
			raise ValueError(f"{points.shape = } not matched at dim 0, which need {dim}")

		self.points = points
		self._dim = dim

	@property
	def n(self):
		return self._dim

	def subsample(self, ratio):
		"""使用随机采样法下采样。
		Args:
			ratio (float)

		TODO: 使用其他采样方法
		"""
		selectedID = np.random.choice(np.arange(0, self.n()), size=int(self.n() * ratio))
		self.points = self.points[:, selectedID]

	# def remove_close(self, radius):
	# 	"""移除距离原点过近的点
	# 	"""
	# 	x = np.abs(self.points[0, :]) < radius
	# 	y = np.abs(self.points[1, :]) < radius
	# 	not_close = np.logical_not(np.logical_and(x, y))
	# 	self.points = self.points[:, not_close]

	def box_cloud(self, box: Box):
		"""generate the BoxCloud for the given pc and box
		Returns:
			[N, 9]: The distance between each point and the box points
		"""
		corner = box.corners()  # [3, 8]
		center = box.center.reshape(-1, 1)  # [3, 1]
		boxPoints = np.concatenate([center, corner], axis=1)  # [3, 9]
		return torch.cdist(self.points.T, boxPoints.T)  # [N, 9]

	def points_in_box(self, box: Box, returnMask=False):
		"""给定一个 Bounding box，返回在这个 box 内的点
		Returns:
			WaterScene_PointCloud / KITTI_PointCloud: 在 box 内的点云，未进行 normalize
			Optional[Tensor[n]]: 返回一个 bool 向量，在 box 内的 point id 为 true
		"""
		newPoints = copy.deepcopy(self)
		newBox    = copy.deepcopy(box)

		rotMat = box.rotation_matrix
		trans  = box.center

		newBox.translate(-trans)
		newPoints.translate(-trans)
		newBox.rotate(Quaternion(matrix=(np.transpose(rotMat))))
		newPoints.rotate(np.transpose(rotMat))

		# maxi = np.max(newBox.corners(), 1)
		# mini = np.min(newBox.corners(), 1)
		maxi =  newBox.wlh / 2
		mini = -newBox.wlh / 2

		x1 = newPoints.points[0, :] < maxi[0]
		x2 = newPoints.points[0, :] > mini[0]
		y1 = newPoints.points[1, :] < maxi[1]
		y2 = newPoints.points[1, :] > mini[1]
		z1 = newPoints.points[2, :] < maxi[2]
		z2 = newPoints.points[2, :] > mini[2]

		includeIDs = np.logical_and(x1, x2)
		includeIDs = np.logical_and(includeIDs, y1)
		includeIDs = np.logical_and(includeIDs, y2)
		includeIDs = np.logical_and(includeIDs, z1)
		includeIDs = np.logical_and(includeIDs, z2)

		dataName = newPoints.__class__
		pointInBox = dataName(newPoints.points[:, includeIDs])
		pointInBox.rotate(rotMat)
		pointInBox.translate(trans)
		if returnMask:
			return pointInBox, pointInBox.box_cloud(box), includeIDs
		return pointInBox, pointInBox.box_cloud(box)

	def convert2Tensor(self):
		return torch.from_numpy(self.points)

	@classmethod
	def fromTensor(cls, tensor):
		points = tensor.numpy()
		return cls(points, points.shape[0])

################################## affine ###################################

	def translate(self, x):
		for i in range(3):
			self.points[i, :] = self.points[i, :] + x[i]

	def rotate(self, rot_mat):
		self.points[:3, :] = np.dot(rot_mat, self.points[:3, :])

	def transform(self, trans_mat):
		self.points[:3, :] = trans_mat.dot(
			np.vstack(
				(self.points[:3, :], np.ones(self.n()))
			)
		)

	# def normalize(self, wlh):
	# 	normalizer = [wlh[1], wlh[0], wlh[2]]
	# 	self.points[:3, :] = self.points[:3, :] / np.atleast_2d(normalizer).T


