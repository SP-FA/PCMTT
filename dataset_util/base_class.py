import copy
import numpy as np
import torch
import torch.nn.functional as F
from pyquaternion import Quaternion

from dataset_util.box_struct import Box


class BaseDataset:
	def __init__(self, cfg):
		self._path = cfg.path
		self._split = cfg.split
		self._coordinate_mode = cfg.coordinate_mode
		self._preload_offset = cfg.preload_offset
		self._preloading = cfg.preloading

	@property
	def num_scenes(self):
		raise NotImplementedError

	@property
	def num_trajectory(self):
		raise NotImplementedError

	@property
	def num_frames(self):
		raise NotImplementedError

	def num_frames_trajectory(self, trajID):
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

	def __repr__(self):
		return f"{self.__class__}  shape: [{self.dim}, {self.n}]"

	@property
	def dim(self):
		return self._dim

	@property
	def n(self):
		return self.points.shape[1]

	def regularize(self, size):
		"""使用随机采样法标准化。
		Args:
			size (int)

		TODO: 使用其他采样方法
		"""
		selectIDs = None
		if self.n > 2:
			if self.n != size:
				selectIDs = np.random.choice(self.n, size=size, replace=size > self.n)
			else:
				selectIDs = np.arange(self.n)
		if selectIDs is not None:
			newPoints = self.points[:, selectIDs]
		else:
			newPoints = np.zeros((self._dim, size), dtype='float32')
		dataName = self.__class__
		points = dataName(newPoints)
		return points, selectIDs

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
		pp = self.points[:3, :].T  # [3, n]
		pp = torch.tensor(pp)
		boxPoints = boxPoints.T  # [3, 9]
		boxPoints = torch.tensor(boxPoints)
		return torch.cdist(pp, boxPoints)

	def points_in_box(self, box: Box, returnMask=False):
		"""给定一个 Bounding box，返回在这个 box 内的点
		Returns:
			WaterScene_PointCloud | KITTI_PointCloud: 在 box 内的点云，未进行 normalize
			Tensor[N, 9]: box cloud 向量，对于每个点，返回相对于 box 8 个 corner + 中心点 的距离
			Optional[Tensor[N]]: 返回一个 bool 向量，在 box 内的 point id 为 true
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

		x1 = newPoints.points[0, :] <= maxi[0]
		x2 = newPoints.points[0, :] >= mini[0]
		y1 = newPoints.points[1, :] <= maxi[1]
		y2 = newPoints.points[1, :] >= mini[1]
		z1 = newPoints.points[2, :] <= maxi[2]
		z2 = newPoints.points[2, :] >= mini[2]

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

	def normalize(self):
		return F.normalize(self.points, dim=-1)

	def convert2Tensor(self):
		return torch.from_numpy(self.points)

	@classmethod
	def fromTensor(cls, tensor):
		points = tensor.numpy()
		return cls(points, points.shape[0])

	def translate(self, x):
		for i in range(3):
			self.points[i, :] = self.points[i, :] + x[i]

	def rotate(self, rot_mat):
		self.points[:3, :] = np.dot(rot_mat, self.points[:3, :])

	def transform(self, trans_mat):
		# 点之间距离可能发生变化
		self.points[:3, :] = trans_mat.dot(
			np.vstack(
				(self.points[:3, :], np.ones(self.n()))
			)
		)
