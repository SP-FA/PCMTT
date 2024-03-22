import torch


def get_one_neighbor(x, points):
    """
        Args:
            x (Tensor[1, 3]): x, y, z
            points (Tensor[n, 3])

        Return:
            Tensor[1, n]
        """
    dist = torch.cdist(x, points)
    neighborID = torch.argmin(dist)
    return neighborID


def get_angle(x, points, useRad=False):
    dotProduct = points @ x.view((-1, 1))  # [n, 1]

    normX = torch.norm(x)  # [1]
    normPoints = torch.norm(points, dim=1).view(-1, 1)  # [n, 1]

    cos = dotProduct / (normPoints * normX)
    angleRad  = torch.acos(cos.T)  # [n]
    if useRad:
        return angleRad.view(-1)
    angleDeg = angleRad * (180.0 / torch.tensor(3.141592653589793))
    return angleDeg.view(-1)


def get_neighbors(x, points, n, theta, alpha=1.25, useRad=False):
    """获取邻域点坐标
    Args:
        x (Tensor[1, 3]): x, y, z
        points (Tensor[n, 3]): 不包含 x
        n (int): 选取几个点
        theta (float): 在 theta 度以内的点排除不选
        useRad (bool): 是否使用弧度制

    Return:
        Tensor[int]: neighbors 的 ID
    """
    neighborsID = []
    for i in range(n):
        neighbor = get_one_neighbor(x, points)
        neighborsID.append(neighbor)

        inThetaIDs = get_angle(points[neighbor] - x, points - x, useRad) > theta
        inRangeIDs = torch.cdist(x, points).view(-1) > alpha * torch.cdist(x, points[neighbor].view(1, -1)).view(-1)
        excludeIDs = torch.logical_and(inThetaIDs, inRangeIDs)
        if torch.all(excludeIDs == False):
            break
        points = points[excludeIDs]
    return torch.Tensor(neighborsID).long()



