import torch
import torch.nn as nn


class PiecewiseNonReferenceLoss(nn.Module):
    """
    Rebuilt from the paper equations.

    Important note:
    The paper gives the functional form, thresholds, and patch size, but does
    not release the exact implementation details for all scalar weights.
    Therefore `m`, `w1`, and `w2` are made configurable in training.
    """

    def __init__(
        self,
        patch_size=16,
        target_value=0.6,
        q1=0.2,
        q2=0.8,
        m=5.0,
        w1=1.0,
        w2=1.0,
    ):
        super(PiecewiseNonReferenceLoss, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.target_value = float(target_value)
        self.q1 = float(q1)
        self.q2 = float(q2)
        self.m = float(m)
        self.w1 = float(w1)
        self.w2 = float(w2)

    def forward(self, input_image, enhanced_image):
        input_gray = torch.mean(input_image, dim=1, keepdim=True)
        enhanced_gray = torch.mean(enhanced_image, dim=1, keepdim=True)

        input_regions = self.pool(input_gray)
        enhanced_regions = self.pool(enhanced_gray)

        sq_error = torch.pow(enhanced_regions - self.target_value, 2)
        extreme_mask = ((input_regions <= self.q1) | (input_regions >= self.q2)).float()
        general_mask = 1.0 - extreme_mask

        extreme_count = torch.clamp(extreme_mask.sum(), min=1.0)
        general_count = torch.clamp(general_mask.sum(), min=1.0)

        l1 = (sq_error * extreme_mask).sum() / extreme_count
        l2 = ((sq_error / (1.0 + self.m * input_regions)) * general_mask).sum() / general_count
        total_loss = self.w1 * l1 + self.w2 * l2

        stats = {
            "piecewise_total": float(total_loss.detach().cpu()),
            "l1_extreme": float(l1.detach().cpu()),
            "l2_general": float(l2.detach().cpu()),
            "extreme_ratio": float(extreme_mask.mean().detach().cpu()),
            "enhanced_region_mean": float(enhanced_regions.mean().detach().cpu()),
            "input_region_mean": float(input_regions.mean().detach().cpu()),
        }
        return total_loss, stats


class L_TV(nn.Module):
    def __init__(self, tv_weight=1.0):
        super(L_TV, self).__init__()
        self.tv_weight = tv_weight

    def forward(self, x):
        batch_size = x.size(0)
        height = x.size(2)
        width = x.size(3)
        count_h = max(1, (height - 1) * width)
        count_w = max(1, height * (width - 1))
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, : height - 1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, : width - 1], 2).sum()
        return self.tv_weight * 2.0 * (h_tv / count_h + w_tv / count_w) / batch_size
