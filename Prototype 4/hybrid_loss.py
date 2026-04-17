import torch
import torch.nn as nn
import torch.nn.functional as F


def rgb_to_gray(x):
    weights = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return torch.sum(x * weights, dim=1, keepdim=True)


def gradient_x(x):
    return x[:, :, :, 1:] - x[:, :, :, :-1]


def gradient_y(x):
    return x[:, :, 1:, :] - x[:, :, :-1, :]


class ColorConstancyLoss(nn.Module):
    def forward(self, x):
        mean_rgb = torch.mean(x, dim=(2, 3), keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        drg = torch.pow(mr - mg, 2)
        drb = torch.pow(mr - mb, 2)
        dgb = torch.pow(mb - mg, 2)
        return torch.mean(torch.sqrt(drg * drg + drb * drb + dgb * dgb + 1e-12))


class SpatialConsistencyLoss(nn.Module):
    def __init__(self, pool_size=4):
        super(SpatialConsistencyLoss, self).__init__()
        self.pool = nn.AvgPool2d(pool_size)
        kernel_left = torch.tensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_right = torch.tensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_up = torch.tensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_down = torch.tensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("weight_left", kernel_left)
        self.register_buffer("weight_right", kernel_right)
        self.register_buffer("weight_up", kernel_up)
        self.register_buffer("weight_down", kernel_down)

    def forward(self, enhanced, original):
        original_mean = self.pool(torch.mean(original, dim=1, keepdim=True))
        enhanced_mean = self.pool(torch.mean(enhanced, dim=1, keepdim=True))

        org_left = F.conv2d(original_mean, self.weight_left, padding=1)
        org_right = F.conv2d(original_mean, self.weight_right, padding=1)
        org_up = F.conv2d(original_mean, self.weight_up, padding=1)
        org_down = F.conv2d(original_mean, self.weight_down, padding=1)

        enh_left = F.conv2d(enhanced_mean, self.weight_left, padding=1)
        enh_right = F.conv2d(enhanced_mean, self.weight_right, padding=1)
        enh_up = F.conv2d(enhanced_mean, self.weight_up, padding=1)
        enh_down = F.conv2d(enhanced_mean, self.weight_down, padding=1)

        loss = (
            torch.pow(org_left - enh_left, 2)
            + torch.pow(org_right - enh_right, 2)
            + torch.pow(org_up - enh_up, 2)
            + torch.pow(org_down - enh_down, 2)
        )
        return torch.mean(loss)


class ExposureControlLoss(nn.Module):
    def __init__(self, patch_size=16, mean_val=0.6):
        super(ExposureControlLoss, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        mean = self.pool(torch.mean(x, dim=1, keepdim=True))
        target = torch.full_like(mean, self.mean_val)
        return torch.mean(torch.pow(mean - target, 2))


class TotalVariationLoss(nn.Module):
    def forward(self, x):
        batch_size = x.size(0)
        h_x = x.size(2)
        w_x = x.size(3)
        count_h = max((h_x - 1) * w_x, 1)
        count_w = max(h_x * (w_x - 1), 1)
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, : h_x - 1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, : w_x - 1], 2).sum()
        return 2.0 * (h_tv / count_h + w_tv / count_w) / max(batch_size, 1)


class EdgeAwareIlluminationLoss(nn.Module):
    def forward(self, illumination, image):
        gray = rgb_to_gray(image)
        weight_x = torch.exp(-10.0 * torch.abs(gradient_x(gray)))
        weight_y = torch.exp(-10.0 * torch.abs(gradient_y(gray)))
        illum_grad_x = torch.abs(gradient_x(illumination))
        illum_grad_y = torch.abs(gradient_y(illumination))
        return torch.mean(illum_grad_x * weight_x) + torch.mean(illum_grad_y * weight_y)


class ReflectanceGradientLoss(nn.Module):
    def forward(self, reflectance_clean, reflectance_raw):
        clean_gray = rgb_to_gray(reflectance_clean)
        raw_gray = rgb_to_gray(reflectance_raw)
        loss_x = F.l1_loss(gradient_x(clean_gray), gradient_x(raw_gray))
        loss_y = F.l1_loss(gradient_y(clean_gray), gradient_y(raw_gray))
        return loss_x + loss_y


class ResidualRegularizationLoss(nn.Module):
    def __init__(self):
        super(ResidualRegularizationLoss, self).__init__()
        self.tv = TotalVariationLoss()

    def forward(self, residual, illumination):
        illumination_rgb = illumination.repeat(1, residual.size(1), 1, 1)
        bright_region_penalty = torch.mean(torch.abs(residual) * illumination_rgb)
        return bright_region_penalty + 0.5 * self.tv(residual)


class HybridZeroDCELoss(nn.Module):
    def __init__(
        self,
        exp_patch_size=16,
        exp_mean_val=0.6,
        weight_color=5.0,
        weight_spa=1.0,
        weight_exp=10.0,
        weight_curve_tv=200.0,
        weight_reconstruction=3.0,
        weight_base_identity=1.0,
        weight_illumination=0.25,
        weight_reflectance_grad=0.2,
        weight_residual=0.05,
    ):
        super(HybridZeroDCELoss, self).__init__()
        self.color_loss = ColorConstancyLoss()
        self.spa_loss = SpatialConsistencyLoss()
        self.exp_loss = ExposureControlLoss(exp_patch_size, exp_mean_val)
        self.tv_loss = TotalVariationLoss()
        self.illumination_loss = EdgeAwareIlluminationLoss()
        self.reflectance_grad_loss = ReflectanceGradientLoss()
        self.residual_loss = ResidualRegularizationLoss()

        self.weight_color = weight_color
        self.weight_spa = weight_spa
        self.weight_exp = weight_exp
        self.weight_curve_tv = weight_curve_tv
        self.weight_reconstruction = weight_reconstruction
        self.weight_base_identity = weight_base_identity
        self.weight_illumination = weight_illumination
        self.weight_reflectance_grad = weight_reflectance_grad
        self.weight_residual = weight_residual

    def forward(self, outputs, input_image):
        enhanced = outputs["enhanced_image"]
        curve_params = outputs["curve_params"]
        illumination = outputs["illumination"]
        reflectance = outputs["reflectance"]
        reflectance_clean = outputs["reflectance_clean"]
        denoise_residual = outputs["denoise_residual"]
        decomposed_image = outputs["decomposed_image"]
        base_image = outputs["base_image"]

        bright_weight = illumination.repeat(1, 3, 1, 1)

        loss_dict = {
            "curve_tv": self.weight_curve_tv * self.tv_loss(curve_params),
            "spa": self.weight_spa * self.spa_loss(enhanced, input_image),
            "color": self.weight_color * self.color_loss(enhanced),
            "exp": self.weight_exp * self.exp_loss(enhanced),
            "reconstruction": self.weight_reconstruction * F.l1_loss(decomposed_image, input_image),
            "base_identity": self.weight_base_identity * torch.mean(torch.abs(base_image - input_image) * bright_weight),
            "illumination": self.weight_illumination * self.illumination_loss(illumination, input_image),
            "reflectance_grad": self.weight_reflectance_grad * self.reflectance_grad_loss(reflectance_clean, reflectance),
            "residual": self.weight_residual * self.residual_loss(denoise_residual, illumination),
        }

        total_loss = sum(loss_dict.values())
        return total_loss, loss_dict
