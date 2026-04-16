import torch
import torch.nn as nn


class enhance_net_nopool(nn.Module):
    """
    Paper-style Zero-DiDCE network rebuilt on top of the original Zero-DCE file name.

    Compared with original Zero-DCE:
    - input image and inverted image are both used
    - the curve is no longer fixed 8-step LE-curve
    - the final enhancement uses paper-described ALE-curve with dynamic
      iterator and amplitude controller
    """

    def __init__(self):
        super(enhance_net_nopool, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        number_f = 32
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f * 2, 3, 3, 1, 1, bias=True)

    def _iteration_count(self, mean_value):
        if mean_value < 0.1:
            count = -25.0 * mean_value + 10.0
        elif mean_value < 0.45:
            count = 17.14 * mean_value * mean_value - 15.14 * mean_value + 10.0
        else:
            count = 5.66 * mean_value * mean_value - 2.93 * mean_value + 7.2
        return max(1, int(count))

    def _beta_value(self, mean_value):
        return -0.79 * mean_value * mean_value + 0.81 * mean_value + 1.41

    def forward(self, x):
        inverse_x = 1.0 - x

        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        curve_a = torch.tanh(self.e_conv7(torch.cat([x1, x3], 1)))

        y1 = self.relu(self.e_conv1(inverse_x))
        y2 = self.relu(self.e_conv2(y1))
        y3 = self.relu(self.e_conv3(y2))
        curve_b = torch.tanh(self.e_conv7(torch.cat([y1, y3], 1)))

        curve_map = (curve_a + curve_b) / 2.0

        alpha = 0.63
        input_mean = float(torch.mean(x).detach().cpu())
        beta = self._beta_value(input_mean)
        num_iter = self._iteration_count(input_mean)

        for _ in range(num_iter):
            current_mean = float(torch.mean(x).detach().cpu())
            denominator = beta - current_mean
            if abs(denominator) < 1e-6:
                denominator = 1e-6 if denominator >= 0 else -1e-6
            gain = (alpha - current_mean) / denominator
            x = x + curve_map * (torch.pow(x, 2) - x) * gain

        meta = {
            "input_mean": input_mean,
            "beta": float(beta),
            "alpha": alpha,
            "num_iter": int(num_iter),
        }
        return x, curve_map, meta


