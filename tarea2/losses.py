import torch
import torch.nn as nn
import torch.nn.functional as F


def crossentropy_loss(y_true, y_pred):
    """
    This is the classical categorical crossentropy
    """
    ce = nn.CrossEntropyLoss()
    out = ce(y_true, y_pred)
    return out


def triplet_loss(margin=20):
    def loss(e_a, e_p, e_n):
        dp = F.pairwise_distance(e_a, e_p, p=2)
        dn = F.pairwise_distance(e_a, e_n, p=2)
        max_ = F.relu(dp - dn + margin)
        return torch.mean(max_)
    return loss


def crossentropy_triplet_loss(y_true_a, y_true_p, y_true_n, y_pred_a, y_pred_p, y_pred_n):
    ce = nn.CrossEntropyLoss()
    ce_a = ce(torch.squeeze(y_true_a), torch.squeeze(y_pred_a))
    ce_p = ce(torch.squeeze(y_true_p), torch.squeeze(y_pred_p))
    ce_n = ce(torch.squeeze(y_true_n), torch.squeeze(y_pred_n))
    loss = (ce_a + ce_p + ce_n) / 3.0
    return loss


def contrastive_loss(margin=1.4):
    def loss(out1, out2, target):
        dist = F.pairwise_distance(out1, out2, p=2)
        # dist_hard = torch.mean(dist)
        # hardest negative and hardest positive
        max_ = F.relu(margin - dist)
        max_squared = torch.square(max_)
        pairwise_contrastive_losses = target * torch.square(dist) + (1 - target) * max_squared
        return torch.mean(pairwise_contrastive_losses)

    return loss


def crossentropy_contrastive_loss(y_true1, y_true2, y_pred1, y_pred2):
    ce = nn.CrossEntropyLoss()
    ce1 = ce(torch.squeeze(y_true1), torch.squeeze(y_pred1))
    ce2 = ce(torch.squeeze(y_true2), torch.squeeze(y_pred2))
    loss = (ce1 + ce2) / 2.0
    return loss
