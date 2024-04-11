import copy

import numpy as np
from pyquaternion import Quaternion


class Box:
    """存储 3d box，包括 label, score and velocity
    """

    def __init__(self, center, size, theta, orient, label=np.nan, score=np.nan,
                 veloc=(np.nan, np.nan, np.nan)):  # , name=None
        """
        Args:
            center (Tuple(float, float, float)): Center of box, x, y, z
            size (Tuple(float, float, float)): width, length, height
            theta (float): rad
            orient: (Quaternion)
            label (int): label, optional
            score (float): Classification score, optional
            veloc: (Tuple(float, float, float)): Box velocity, x, y, z direction.
            # name (str): Box name, optional.
        """
        assert not np.any(np.isnan(center))
        assert not np.any(np.isnan(size))
        assert len(center) == 3
        assert len(size) == 3

        self.center = np.array(center)
        self.wlh = np.array(size)
        self.theta = theta
        self.orient = orient
        self.label = int(label) if not np.isnan(label) else label
        self.score = float(score) if not np.isnan(score) else score
        self.veloc = np.array(veloc)
        # self.name  = name

    def __eq__(self, other):
        center = np.allclose(self.center, other.center)
        wlh = np.allclose(self.wlh, other.wlh)
        orient = np.allclose(self.orient.elements, other.orient.elements)
        label = (self.label == other.label) or (np.isnan(self.label) and np.isnan(other.label))
        score = (self.score == other.score) or (np.isnan(self.score) and np.isnan(other.score))
        veloc = (np.allclose(self.veloc, other.veloc) or
                 (np.all(np.isnan(self.veloc)) and np.all(np.isnan(other.veloc))))
        return center and wlh and orient and label and score and veloc

    def __repr__(self):
        repr_str = 'label: {}, score: {:.2f}, xyz: [{:.2f}, {:.2f}, {:.2f}], wlh: [{:.2f}, {:.2f}, {:.2f}], ' \
                   'rot axis: [{:.2f}, {:.2f}, {:.2f}], ang(degrees): {:.2f}, ang(rad): {:.2f}, ' \
                   'vel: {:.2f}, {:.2f}, {:.2f}'
        return repr_str.format(self.label, self.score,
                               self.center[0], self.center[1], self.center[2],
                               self.wlh[0], self.wlh[1], self.wlh[2],
                               self.orient.axis[0], self.orient.axis[1], self.orient.axis[2], self.orient.degrees,
                               self.orient.radians,
                               self.veloc[0], self.veloc[1], self.veloc[2])

    def encode(self):
        """Encodes the box instance to a JSON-friendly vector representation.

        Returns:
            List[float * 16]
        """
        return self.center.tolist() + self.wlh.tolist() + self.orient.elements.tolist() + [self.label] + [
            self.score] + self.veloc.tolist()

    @classmethod
    def decode(cls, data):
        return Box(data[0:3], data[3:6], Quaternion(data[6:10]), label=data[10], score=data[11], veloc=data[12:15])

    @property
    def rotation_matrix(self):
        return self.orient.rotation_matrix

    def corners(self, wlh_factor=1.0):
        """返回 bbox 的角坐标

        Returns:
            <np.array[3, 8], np.float>: 前四个坐标表示面向前方的四个角
        """
        w, l, h = self.wlh * wlh_factor

        x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))
        corners = np.dot(self.orient.rotation_matrix, corners)

        x, y, z = self.center
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z
        return corners

    def bottom_corners(self):
        return self.corners()[:, [2, 3, 7, 6]]

    ################################## affine ###################################

    def translate(self, x):
        self.center += x

    def rotate(self, quaternion):
        self.center = np.dot(quaternion.rotation_matrix, self.center)
        self.orient = quaternion * self.orient
        self.veloc = np.dot(quaternion.rotation_matrix, self.veloc)

    def scale(self, ratio):
        self.wlh *= ratio

    def get_offset_box(self, offset, ratio=1):
        """根据 offset 和 ratio 调整 box 并返回调整后的 box

        Args:
            offset (List[float * 3]): x, y 的偏移和绕 z 旋转的角度
        """
        new_box = copy.deepcopy(self)
        new_box.scale(ratio)
        new_box.translate([offset[0], offset[1], 0])
        new_box.rotate(Quaternion(axis=[0, 0, 1], radians=offset[2] * np.deg2rad(5)))
        return new_box

    def transform(self, trans_mat):
        transformed = np.dot(trans_mat[0:3, 0:4].T, self.center)
        self.center = transformed[0:3] / transformed[3]
        self.orient = self.orient * Quaternion(matrix=trans_mat[0:3, 0:3])
        self.veloc = np.dot(trans_mat[0:3, 0:3], self.veloc)