import unittest
import torch
import time
from model.DGN.graph_construction import GraphConstructor


class MyTestCase(unittest.TestCase):
    def test_new_construction_time(self):
        n = 1000
        start = time.time()
        for i in range(10):
            points = torch.randn(n, 3, dtype=torch.float)
            gc = GraphConstructor(points, 5, 20)
            edges = gc.get_edges()
            print(edges)

        print(time.time() - start)


if __name__ == '__main__':
    unittest.main()
