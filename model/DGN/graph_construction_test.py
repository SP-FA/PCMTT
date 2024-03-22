import unittest
import torch

from graph_construction import get_one_neighbor, get_angle, get_neighbors


class MyTestCase(unittest.TestCase):
    def test_get_one_neighbor(self):
        points = torch.randn(100, 3, dtype=torch.float)
        print(points.shape)
        x = torch.tensor([0, 0, 0], dtype=torch.float).view((1, -1))
        print(x.shape)

        nb = get_one_neighbor(x, points)
        print(points[nb])

    def test_get_angle(self):
        points = torch.randn(100, 3, dtype=torch.float)
        print(points.shape)
        x = torch.tensor([1, 1, 1], dtype=torch.float).view((1, -1))
        print(x.shape)

        a = get_angle(x, points)
        print(a.shape)

        a = get_angle(x, points, True)
        print(a.shape)

    def test_get_neighbors(self):
        points = torch.randn(1000, 3, dtype=torch.float)
        print(points.shape)
        x = torch.tensor([1, 1, 1], dtype=torch.float).view((1, -1))
        print(x.shape)

        ns = get_neighbors(x, points, 5, 20, 5)
        print(ns.shape)
        print(points[ns])


if __name__ == '__main__':
    unittest.main()
