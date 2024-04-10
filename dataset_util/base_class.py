import copy
import numpy as np
import torch
import torch.nn.functional as F
from pyquaternion import Quaternion

from dataset_util.box_struct import Box


class BaseDataset:
	def __init__(self, cfg, split):
		self._path = cfg.path
		self._split = split
		self._coordinate_mode = cfg.coordinate_mode
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
