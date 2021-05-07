import torch
import torch.nn as nn
def triplet_loss(margin = 20):
    def loss(y_true, y_pred):
        #y_true will be used for training cross_entropy
        e_a, e_p, e_n = torch.split(y_pred, split_size_or_sections = 3, dim = 1)
        d_p = torch.sqrt(torch.sum(torch.square(e_a - e_p), 2))
        d_p_hard = torch.mean(d_p)
        d_n = torch.sqrt(torch.sum(torch.square(e_a - e_n), 2))
        d_n_hard = torch.mean(d_n)
        #hardest negative and hardest positive
        _, max = torch.max(1e-10, d_p_hard + margin - d_n_hard)
        return max
    return loss

def crossentropy_loss(y_true, y_pred):
    """
    This is the classical categorical crossentropy
    """
    ce = nn.CrossEntropyLoss()
    out = ce(y_true, y_pred, from_logits=True)
    return out

def crossentropy_triplet_loss(y_true, y_pred):
    y_true_a, y_true_p, y_true_n = torch.split(y_true, split_size_or_sections = 3, dim = 1)
    cl_a, cl_p, cl_n = torch.split(y_pred, split_size_or_sections = 3, dim = 1)
    ce = nn.CrossEntropyLoss()
    ce_a = ce(torch.squeeze(y_true_a), torch.squeeze(cl_a), from_logits=True)
    ce_p = ce(torch.squeeze(y_true_p), torch.squeeze(cl_p), from_logits=True)
    ce_n = ce(torch.squeeze(y_true_n), torch.squeeze(cl_n), from_logits=True)
    loss = (ce_a + ce_p + ce_n) / 3.0
    return loss