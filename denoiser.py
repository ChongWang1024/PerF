import torch
from torch import nn


class Denoiser(nn.Module):
    def __init__(
        self,
        net,
        *,
        label_drop_prob=0.1,
        P_mean=-0.8,
        P_std=0.8,
        t_eps=0.05,
        noise_scale=1.0,
        cfg_scale=1.0,
        pg_scale=1.0,
        cfg_interval=(0.0, 1.0),
        pg_interval=(0.0, 1.0),
        sampling_method="heun",
        num_sampling_steps=50,
    ):
        super().__init__()
        self.net = net
        self.label_drop_prob = label_drop_prob
        self.P_mean, self.P_std = P_mean, P_std
        self.t_eps, self.noise_scale = t_eps, noise_scale
        self.cfg_scale, self.pg_scale = cfg_scale, pg_scale
        self.cfg_interval, self.pg_interval = cfg_interval, pg_interval
        self.method, self.steps = sampling_method, num_sampling_steps
        if self.method not in ("heun", "euler") or self.steps < 1:
            raise ValueError("Use heun/euler and a positive number of sampling steps")
        for interval in (cfg_interval, pg_interval):
            if not 0 <= interval[0] < interval[1] <= 1:
                raise ValueError("Guidance intervals must satisfy 0 <= low < high <= 1")
        if not 0 <= label_drop_prob <= 1 or t_eps <= 0 or noise_scale <= 0:
            raise ValueError("Invalid label dropout, t_eps or noise scale")

    def forward(self, x, labels, return_persistent=False):
        if self.training:
            drop = (
                torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
            )
            labels = torch.where(drop, self.net.num_classes, labels)
        t = (
            torch.randn(x.shape[0], device=x.device) * self.P_std + self.P_mean
        ).sigmoid()
        t = t.view(-1, 1, 1, 1)
        e = torch.randn_like(x) * self.noise_scale
        z = t * x + (1 - t) * e
        prediction = self.net(
            z, t.flatten(), labels, return_persistent=return_persistent
        )
        if return_persistent:
            prediction, features = prediction
        velocity = (x - z) / (1 - t).clamp_min(self.t_eps)
        predicted_velocity = (prediction - z) / (1 - t).clamp_min(self.t_eps)
        loss = (velocity - predicted_velocity).square().mean(dim=(1, 2, 3)).mean()
        return (loss, features) if return_persistent else loss

    @staticmethod
    def interval_scale(t, scale, interval):
        low, high = interval
        active = (t < high) & ((low == 0) | (t > low))
        return torch.where(active, scale, 1.0)

    @torch.no_grad()
    def guided_velocity(self, z, t, labels):
        """Class/on, null-class/on, class/off; PG=1 reduces to ordinary CFG."""

        def predict(y, p_to_a_scale):
            x = self.net(z, t.flatten(), y, p_to_a_scale=p_to_a_scale)
            return (x - z) / (1 - t).clamp_min(self.t_eps)

        v_cond = predict(labels, 1.0)
        v_uncond = predict(torch.full_like(labels, self.net.num_classes), 1.0)
        cfg = self.interval_scale(t, self.cfg_scale, self.cfg_interval)
        velocity = v_uncond + cfg * (v_cond - v_uncond)
        if self.pg_scale != 1.0:
            v_without_p_to_a = predict(labels, 0.0)
            pg = self.interval_scale(t, self.pg_scale, self.pg_interval)
            velocity = velocity + (pg - 1.0) * (v_cond - v_without_p_to_a)
        return velocity

    @torch.no_grad()
    def generate(self, labels, noise=None):
        shape = (
            labels.shape[0],
            self.net.in_channels,
            self.net.input_size,
            self.net.input_size,
        )
        z = (
            self.noise_scale * torch.randn(shape, device=labels.device)
            if noise is None
            else noise
        )
        if tuple(z.shape) != shape:
            raise ValueError(
                f"Expected initial noise shape {shape}, got {tuple(z.shape)}"
            )
        times = torch.linspace(0, 1, self.steps + 1, device=labels.device)
        for i in range(self.steps):
            t = times[i].expand(labels.shape[0]).view(-1, 1, 1, 1)
            t_next = times[i + 1].expand_as(t)
            velocity = self.guided_velocity(z, t, labels)
            proposal = z + (t_next - t) * velocity
            if self.method == "heun" and i < self.steps - 1:
                next_velocity = self.guided_velocity(proposal, t_next, labels)
                z = z + (t_next - t) * (0.5 * (velocity + next_velocity))
            else:
                z = proposal
        return z
