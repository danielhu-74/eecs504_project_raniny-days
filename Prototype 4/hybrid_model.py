import torch
import torch.nn as nn
import torch.nn.functional as F


def rgb_to_gray(x):
    weights = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return torch.sum(x * weights, dim=1, keepdim=True)


class HybridZeroDCE(nn.Module):
    def __init__(self, base_channels=32, curve_steps=8, eps=1e-4):
        super(HybridZeroDCE, self).__init__()
        self.base_channels = base_channels
        self.curve_steps = curve_steps
        self.eps = eps
        self.relu = nn.ReLU(inplace=True)

        self.d_conv1 = nn.Conv2d(3, base_channels, 3, 1, 1, bias=True)
        self.d_conv2 = nn.Conv2d(base_channels, base_channels, 3, 1, 1, bias=True)
        self.d_conv3 = nn.Conv2d(base_channels, base_channels, 3, 1, 1, bias=True)
        self.d_conv4 = nn.Conv2d(base_channels, base_channels, 3, 1, 1, bias=True)
        self.d_conv5 = nn.Conv2d(base_channels * 2, base_channels, 3, 1, 1, bias=True)
        self.d_conv6 = nn.Conv2d(base_channels * 2, base_channels, 3, 1, 1, bias=True)

        self.illum_conv1 = nn.Conv2d(base_channels * 2, base_channels, 3, 1, 1, bias=True)
        self.illum_conv2 = nn.Conv2d(base_channels, 1, 3, 1, 1, bias=True)

        self.refl_conv1 = nn.Conv2d(base_channels * 2 + 4, base_channels, 3, 1, 1, bias=True)
        self.refl_conv2 = nn.Conv2d(base_channels, base_channels, 3, 1, 1, bias=True)
        self.refl_conv3 = nn.Conv2d(base_channels, 3, 3, 1, 1, bias=True)

        self.e_conv1 = nn.Conv2d(3, base_channels, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(base_channels, base_channels, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(base_channels, base_channels, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(base_channels, base_channels, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(base_channels * 2, base_channels, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(base_channels * 2, base_channels, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(base_channels * 2, curve_steps * 3, 3, 1, 1, bias=True)

    def forward_decomposition(self, x):
        d1 = self.relu(self.d_conv1(x))
        d2 = self.relu(self.d_conv2(d1))
        d3 = self.relu(self.d_conv3(d2))
        d4 = self.relu(self.d_conv4(d3))
        d5 = self.relu(self.d_conv5(torch.cat([d3, d4], dim=1)))
        d6 = self.relu(self.d_conv6(torch.cat([d2, d5], dim=1)))

        shared = torch.cat([d1, d6], dim=1)
        illum_logits = self.illum_conv2(self.relu(self.illum_conv1(shared)))
        lowlight_prior = rgb_to_gray(x)
        illumination = torch.sigmoid(0.7 * illum_logits + 0.3 * torch.logit(lowlight_prior.clamp(0.05, 0.95)))
        illumination = illumination.clamp(0.05, 1.0)

        illumination_rgb = illumination.repeat(1, 3, 1, 1)
        reflectance = torch.clamp(x / (illumination_rgb + self.eps), 0.0, 1.0)

        darkness = 1.0 - illumination_rgb
        refl_context = torch.cat([shared, reflectance, illumination], dim=1)
        denoise_residual = torch.tanh(
            self.refl_conv3(self.relu(self.refl_conv2(self.relu(self.refl_conv1(refl_context)))))
        )
        reflectance_clean = torch.clamp(reflectance + darkness * denoise_residual, 0.0, 1.0)

        decomposed_image = torch.clamp(reflectance_clean * illumination_rgb, 0.0, 1.0)
        base_image = torch.clamp(x + darkness * (decomposed_image - x), 0.0, 1.0)

        return {
            "illumination": illumination,
            "reflectance": reflectance,
            "reflectance_clean": reflectance_clean,
            "denoise_residual": denoise_residual,
            "decomposed_image": decomposed_image,
            "base_image": base_image,
        }

    def forward_enhancer(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], dim=1)))

        curve_params = torch.tanh(self.e_conv7(torch.cat([x1, x6], dim=1)))
        curve_chunks = torch.chunk(curve_params, self.curve_steps, dim=1)

        enhanced = x
        enhanced_mid = None
        for index, curve in enumerate(curve_chunks):
            enhanced = enhanced + curve * (torch.pow(enhanced, 2) - enhanced)
            if index == (self.curve_steps // 2) - 1:
                enhanced_mid = enhanced

        if enhanced_mid is None:
            enhanced_mid = enhanced

        return {
            "enhanced_image_1": torch.clamp(enhanced_mid, 0.0, 1.0),
            "enhanced_image": torch.clamp(enhanced, 0.0, 1.0),
            "curve_params": curve_params,
        }

    def forward(self, x):
        decomposition_outputs = self.forward_decomposition(x)
        enhancement_outputs = self.forward_enhancer(decomposition_outputs["base_image"])
        outputs = {}
        outputs.update(decomposition_outputs)
        outputs.update(enhancement_outputs)
        return outputs

    def load_zero_dce_pretrain(self, checkpoint_path, map_location=None):
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint = checkpoint["model"]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        model_dict = self.state_dict()
        matched_keys = []
        skipped_keys = []
        for key, value in checkpoint.items():
            if key in model_dict and model_dict[key].shape == value.shape:
                model_dict[key] = value
                matched_keys.append(key)
            else:
                skipped_keys.append(key)

        self.load_state_dict(model_dict, strict=False)
        return matched_keys, skipped_keys
