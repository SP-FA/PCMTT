import torch
import numpy as np


def cartesian2polar(points, normRes=True):
    """
    Args:
        points (tensor): [B, N, 3] / [B, N, k, 3] 必须是 normalize 之后的坐标
        normRes (bool): 是否对角度进行归一化

    Return:
        tensor: [B, N, 3] / [B, N, k, 3]
    """
    r = torch.sqrt(torch.sum(torch.pow(points, 2), dim=-1, keepdim=True))
    r = torch.clamp(r, min=0)
    theta = torch.acos(points[..., 2, None] / r)  # range: [0, pi]
    phi = torch.atan2(points[..., 1, None], points[..., 0, None])  # range: [-pi, pi]

    # check nan
    idx = r == 0
    theta[idx] = 0

    if normRes:
        theta = theta / np.pi  # [0, 1]
        phi = phi / (2 * np.pi) + 0.5  # [0, 1]
    return torch.cat([r, theta, phi], dim=-1)


def cal_normal(groupPoints, randInvNormal=False):
    """
    Calculate Normal Vector (Unit Form + First Term Positive)

    Args:
        groupPoints (tensor): [B, N, 3, 3] / [B, N, k-1, 3, 3]
        randInvNormal (bool)

    Return:
        tensor: [B, N, 3] / [B, N, k, 3]
    """
    isGroup = len(groupPoints.shape) == 5

    vec1 = groupPoints[..., 1, :] - groupPoints[..., 0, :]  # [B, N, 3] / [B, N, k, 3]
    vec2 = groupPoints[..., 2, :] - groupPoints[..., 0, :]  # [B, N, 3] / [B, N, k, 3]

    normal = torch.cross(vec1, vec2, dim=-1)
    unitNormal = normal / torch.norm(normal, dim=-1, keepdim=True)  # [B, N, 3] / [B, N, k, 3]
    if isGroup:
        pos_mask = (unitNormal[..., 0:1, 0] > 0).float() * 2. - 1.
    else:
        pos_mask = (unitNormal[..., 0] > 0).float() * 2. - 1.
    unitNormal = unitNormal * pos_mask.unsqueeze(-1)

    if randInvNormal:
        random_mask = torch.randint(0, 2, (groupPoints.size(0), 1, 1)).float() * 2. - 1.
        random_mask = random_mask.to(unitNormal.device)
        if isGroup:
            unitNormal = unitNormal * random_mask.unsqueeze(-1)
        else:
            unitNormal = unitNormal * random_mask

    return unitNormal


def cal_center(groupPoints):
    """
    Args:
        groupPoints (tensor): [B, N, 3, 3] / [B, N, k, 3, 3]
    Return:
        tensor: [B, N, 3] / [B, N, k, 3]
    """
    return torch.mean(groupPoints, dim=-2)


def cal_constant(normal, center, normalize=True):
    """
    Calculate Constant Term (Standard Version, with x_normal to be 1)
        const = x_normal * x_0 + y_normal * y_0 + z_normal * z_0

    Args:
        normal (tensor): [B, N, 3] / [B, N, G, 3]
        center (tensor): [B, N, 3] / [B, N, G, 3]
        normalize (bool)

    Return:
        tensor: [B, N, 1] / [B, N, G, 1]
    """
    const = torch.sum(normal * center, dim=-1, keepdim=True)
    if normalize:
        factor = torch.sqrt(torch.Tensor([3])).to(normal.device)
        const = const / factor
    return const


def check_nan(normal, center, pos=None):
    """
    Check & Remove NaN in normal tensor

    Args:
        normal (tensor): [B, N, k, 3]
        center (tensor): [B, N, k, 3]
        pos (tensor): [B, N, k, 1]

    Return:
        tensor: [B, N, k, 3] normal without nan
        tensor: [B, N, k, 3] center
        tensor[Optional]: [B, N, k, 1] position
    """
    B, N, k, _ = normal.shape
    mask = torch.sum(torch.isnan(normal), dim=-1) > 0
    tempValue = torch.argmax((~mask).int(), dim=-1)
    idxB = torch.arange(B).unsqueeze(1).repeat([1, N])
    idxN = torch.arange(N).unsqueeze(0).repeat([B, 1])

    normalTemp = normal[idxB, idxN, None, tempValue].repeat([1, 1, k, 1])
    normal[mask] = normalTemp[mask]
    centerTemp = center[idxB, idxN, None, tempValue].repeat([1, 1, k, 1])
    center[mask] = centerTemp[mask]

    if pos is not None:
        posTemp = pos[idxB, idxN, None, tempValue].repeat([1, 1, k, 1])
        pos[mask] = posTemp[mask]
        return normal, center, pos
    return normal, center
