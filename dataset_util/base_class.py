import numpy as np
import torch


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

	def convert2Tensor(self):
		return torch.from_numpy(self.points)

	@classmethod
	def fromTensor(cls, tensor):
		points = tensor.numpy()
		return cls(points, points.shape[0])
