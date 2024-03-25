import torch
import torch.nn.functional as F


class GraphConstructor:
    INTMAX = 1e9

    def __init__(self, points, k, theta, alpha=1.25, useRad=False):
        """
        Args:
            points (Tensor[n, 3])
            k (int): 选取几个点
            theta (float): 在 theta 度以内的点排除不选
            alpha (float): 小于 alpha * ||x - y|| 范围内的点不选
            useRad (bool): 是否使用弧度制
        """
        self.points = points
        self.k = k
        self.theta = theta
        self.a = alpha
        self.useRad = useRad
        self.distanceMat = None
        self.dist()

    @property
    def dim(self):
        return self.points.shape[1]

    @property
    def n(self):
        return self.points.shape[0]

    def dist(self):
        if self.distanceMat is None:
            self.distanceMat = torch.cdist(self.points, self.points)
        self.distanceMat = self.distanceMat + torch.eye(self.n) * self.INTMAX

    def get_one_neighbor(self, pid):
        """
        Args:
            pid (int): point ID

        Return:
            Tensor[1]
        """
        dist = self.distanceMat[pid]
        neighborID = torch.argmin(dist)
        return neighborID

    def get_angle(self, neighborID, vectors):
        x = vectors[neighborID].view(1, -1)  # [1, 3]
        cos = F.cosine_similarity(x, vectors, dim=1)  # [n]
        angleRad = torch.acos(cos)
        if self.useRad:
            return angleRad
        angleDeg = torch.rad2deg(angleRad)
        return angleDeg

    def get_neighbors(self, pid):
        """获取邻域点坐标
        Return:
            Tensor[int * k]: neighbors 的 ID
        """
        x = self.points[pid].view(1, -1)
        neighborsID = []
        for i in range(self.k):
            neighbor = self.get_one_neighbor(pid)
            assert neighbor != pid
            neighborsID.append(neighbor)

            inThetaIDs = self.get_angle(neighbor, self.points - x) > self.theta
            inRangeIDs = self.distanceMat[pid] > self.a * self.distanceMat[pid][neighbor]
            includeIDs = torch.logical_and(inThetaIDs, inRangeIDs)
            excludeIDs = torch.logical_not(includeIDs)
            if torch.all(excludeIDs == True):
                break
            self.distanceMat[pid][excludeIDs] = self.INTMAX
        return torch.Tensor(neighborsID).long()

    def get_edges(self):
        edges = []
        for i in range(self.n):
            edge = self.get_neighbors(i)
            edges.append(edge)
        return edges

