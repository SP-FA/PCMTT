import unittest
import numpy as np
from pyquaternion import Quaternion
from dataset_util.box_struct import Box

class MyTestCase(unittest.TestCase):
    def test_Box(self):
        q = Quaternion()
        print(q.rotation_matrix)
        b1 = Box(np.array([1, 1, 1]), np.array([2, 2, 2]), q)
        print(b1)
        b2 = Box(np.array([1, 1, 1]), np.array([2, 2, 2]), q)
        b3 = Box(np.array([0, 1, 1]), np.array([2, 2, 2]), q)
        b4 = Box(np.array([1, 1, 1]), np.array([1, 2, 2]), q)
        b5 = Box(np.array([1, 1, 1]), np.array([2, 2, 2]), q, label=1)
        b6 = Box(np.array([1, 1, 1]), np.array([2, 2, 2]), q, score=2)
        b7 = Box(np.array([1, 1, 1]), np.array([2, 2, 2]), q, veloc=[1, 1, 1])
        assert b1 == b2
        assert b1 != b3
        assert b1 != b4
        assert b1 != b5
        assert b1 != b6
        assert b1 != b7

        e = b1.encode()
        b8 = Box.decode(e)
        assert b1 == b8

        print(b1.corners())
        print(b1.bottom_corners())

    def test_Box_affine(self):
        q = Quaternion()
        print(q.rotation_matrix)
        b1 = Box(np.array([1, 1, 1]), np.array([2, 2, 2]), q)
        b1.translate(np.array([1, 1, 1]))
        print(b1.corners())

if __name__ == '__main__':
    unittest.main()
