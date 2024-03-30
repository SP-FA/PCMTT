import torch


def knn(x, k):
    """
    Args:
        x: [B, C, N]
        k (int)
    """
    xT = x.transpose(2, 1)  # [B, N, C]
    distance = torch.cdist(xT, xT)  # [B, N, N]
    _, idx = distance.topk(k=k, dim=-1)  # [B, N, k]
    return idx


def get_graph_feature(x, k):
    """
    Args:
        x Tensor[B, C, N]: points, C=3
        k

    Returns:
        Tensor[B, N, k, C]
    """
    idx = knn(x, k)  # [B, N, k]
    x = x.transpose(2, 1).contiguous()  # [B, N, C]
    C = x.shape[-1]
    feature = torch.gather(x.unsqueeze(2), 1, idx.unsqueeze(-1).expand(-1, -1, -1, C))  # [B, N, k, C]
    return feature

