"""From https://github.com/raminnakhli/Decoupled-Contrastive-Learning."""

import numpy as np
import torch

SMALL_NUM = np.log(1e-45)


class DCL(torch.nn.Module):
    """
    Decoupled Contrastive Loss proposed in https://arxiv.org/pdf/2110.06848.pdf
    weight: the weighting function of the positive sample loss
    temperature: temperature to control the sharpness of the distribution
    """

    def __init__(self, temperature=0.1, weight_fn=None):
        super().__init__()
        self.temperature = temperature
        self.weight_fn = weight_fn

    def forward(self, out_1_out_2, *args, **kwargs):
        """
        Calculate one way DCL loss
        :return: one-way loss
        """
        orig_size = int(out_1_out_2.shape[0] / 2)
        z1 = out_1_out_2[:orig_size]
        z2 = out_1_out_2[orig_size:]

        cross_view_distance = torch.mm(z1, z2.t())
        positive_loss = -torch.diag(cross_view_distance) / self.temperature
        if self.weight_fn is not None:
            positive_loss = positive_loss * self.weight_fn(z1, z2)
        neg_similarity = torch.cat((torch.mm(z1, z1.t()), cross_view_distance), dim=1) / self.temperature
        neg_mask = torch.eye(z1.size(0), device=z1.device).repeat(1, 2)
        negative_loss = torch.logsumexp(neg_similarity + neg_mask * SMALL_NUM, dim=1, keepdim=False)
        return (positive_loss + negative_loss).mean()


class DCLW(DCL):
    """
    Decoupled Contrastive Loss with negative von Mises-Fisher weighting proposed in https://arxiv.org/pdf/2110.06848.pdf
    sigma: the weighting function of the positive sample loss
    temperature: temperature to control the sharpness of the distribution
    """

    def __init__(self, sigma=0.5, temperature=0.1):
        self.sigma = sigma
        self.temperature = temperature

        super().__init__(weight_fn=self.neg_von_mises_fisher, temperature=temperature)

    def neg_von_mises_fisher(self, z1, z2):
        return 2 - z1.size(0) * torch.nn.functional.softmax((z1 * z2).sum(dim=1) / self.sigma, dim=0).squeeze()
