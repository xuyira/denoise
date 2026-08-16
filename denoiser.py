import torch
import torch.nn as nn
import torch.nn.functional as F

from models.raw_vit import RawViTDiffusion


class Denoiser(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.net = RawViTDiffusion(
            model_name=args.model,
            num_channels=args.num_eeg_channels,
            patch_size=args.eeg_patch_size,
            target_length=args.target_length,
            attn_dropout=args.attn_dropout,
            proj_dropout=args.proj_dropout,
            use_convffn=getattr(args, "use_convffn", False),
            convffn_kernel_size=getattr(args, "convffn_kernel_size", 3),
        )
        self.sample_shape = (args.num_eeg_channels, args.eeg_patch_num, args.eeg_patch_size)
        self.raw_vit_num_channels = args.num_eeg_channels
        self.raw_vit_target_length = args.target_length
        self.raw_vit_patch_size = args.eeg_patch_size
        self.raw_vit_patch_num = args.eeg_patch_num

        self.loss_type = getattr(args, "loss_type", "l2")
        self.loss_weight_recon = getattr(args, "loss_weight_recon", 0.5)
        self.loss_weight_stft = getattr(args, "loss_weight_stft", 0.2)
        self.loss_weight_stat = getattr(args, "loss_weight_stat", 1.0)
        self.loss_weight_tv = getattr(args, "loss_weight_tv", 0.05)
        self.loss_weight_corr = getattr(args, "loss_weight_corr", 0.05)
        self.loss_weight_velocity = getattr(args, "loss_weight_velocity", 0.0)
        self.prediction_target = getattr(args, "prediction_target", "velocity")
        self.clean_output = getattr(args, "clean_output", "direct")
        self.denoise_mode = getattr(args, "denoise_mode", "ode")
        self.t_eps = getattr(args, "t_eps", 1e-5)
        if self.prediction_target not in {"velocity", "clean"}:
            raise ValueError("prediction_target must be 'velocity' or 'clean'.")
        if self.clean_output not in {"direct", "residual"}:
            raise ValueError("clean_output must be 'direct' or 'residual'.")
        if self.denoise_mode not in {"ode", "direct"}:
            raise ValueError("denoise_mode must be 'ode' or 'direct'.")

        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        self.method = args.sampling_method
        self.steps = args.num_sampling_steps

    def sample_t(self, n: int, device=None):
        return torch.rand(n, device=device)

    def to_training_space(self, batch: torch.Tensor) -> torch.Tensor:
        return self._raw_vit_to_patches(batch)

    def _raw_vit_to_patches(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.dim() == 4 and batch.shape[1] == self.raw_vit_num_channels and batch.shape[2] == self.raw_vit_patch_num and batch.shape[3] == self.raw_vit_patch_size:
            return batch
        if batch.dim() == 3 and batch.shape[0] == self.raw_vit_num_channels and batch.shape[1] == self.raw_vit_patch_num and batch.shape[2] == self.raw_vit_patch_size:
            return batch

        added_batch_dim = False
        if batch.dim() == 2:
            batch = batch.unsqueeze(0)
            added_batch_dim = True
        elif batch.dim() == 3 and batch.shape[0] == self.raw_vit_num_channels:
            batch = batch.unsqueeze(0)
            added_batch_dim = True

        if batch.dim() < 3:
            raise ValueError("Expected raw EEG tensors with at least 3 dims (batch, channels, length).")
        if batch.shape[1] != self.raw_vit_num_channels:
            raise ValueError(f"Expected {self.raw_vit_num_channels} EEG channels but received {batch.shape[1]}.")

        total = batch.shape[-1]
        target = self.raw_vit_target_length
        if total < target:
            batch = F.pad(batch, (0, target - total))
        elif total > target:
            batch = batch[..., :target]

        batch = batch.view(batch.shape[0], self.raw_vit_num_channels, self.raw_vit_patch_num, self.raw_vit_patch_size)

        if added_batch_dim:
            batch = batch.squeeze(0)
        return batch

    def forward(self, noisy, clean):
        noisy = self.to_training_space(noisy)
        clean = self.to_training_space(clean)

        t = self.sample_t(clean.size(0), device=clean.device).view(-1, *([1] * (clean.ndim - 1)))
        z = (1 - t) * noisy + t * clean
        v = clean - noisy

        if self.prediction_target == "clean":
            clean_pred = self._clean_pred_from_net(z, t.flatten())
            v_pred = (clean_pred - z) / (1 - t).clamp_min(self.t_eps)
        else:
            net_out = self.net(z, t.flatten())
            v_pred = net_out
            clean_pred = z + (1 - t) * v_pred

        return self._compute_loss(v, v_pred, clean, clean_pred)

    def _clean_pred_from_net(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        net_out = self.net(z, t)
        return z + net_out if self.clean_output == "residual" else net_out

    def _compute_loss(self, v, v_pred, x, x_pred):
        if self.prediction_target == "clean":
            return self._compute_clean_loss(v, v_pred, x, x_pred)

        loss_l1 = (x - x_pred).abs().mean()

        if self.loss_type == "mix":
            mix_loss = ((v - v_pred) ** 2).mean() + self.loss_weight_recon * loss_l1

            if self.loss_weight_stft > 0:
                fft_sizes = [64, 128, 256, 512, 1024]
                hop_sizes = [16, 32, 64, 128, 256]
                win_lengths = [64, 128, 256, 512, 1024]
                signal_length = x.reshape(x.shape[0], x.shape[1], -1).shape[-1]
                stft_configs = [
                    (n_fft, hop, win)
                    for n_fft, hop, win in zip(fft_sizes, hop_sizes, win_lengths)
                    if n_fft <= signal_length
                ]

                loss_stft = x.new_tensor(0.0)
                for n_fft, hop, win in stft_configs:
                    loss_stft += self._stft_loss_multires(x, x_pred, n_fft, hop, win)
                if stft_configs:
                    loss_stft /= len(stft_configs)
                    mix_loss = mix_loss + self.loss_weight_stft * loss_stft

            if self.loss_weight_stat > 0:
                mix_loss = mix_loss + self.loss_weight_stat * self._statistics_loss(x, x_pred)

            if self.loss_weight_tv > 0:
                mix_loss = mix_loss + self.loss_weight_tv * self._total_variation_loss(x_pred)

            if self.loss_weight_corr > 0:
                mix_loss = mix_loss + self.loss_weight_corr * self._pearson_corr_loss(x, x_pred)

            return mix_loss

        if self.loss_type == "l1":
            return (v - v_pred).abs().mean()
        return ((v - v_pred) ** 2).mean()

    def _compute_clean_loss(self, v, v_pred, x, x_pred):
        loss_l1 = (x - x_pred).abs().mean()

        if self.loss_type == "mix":
            loss = loss_l1
            if self.loss_weight_velocity > 0:
                loss = loss + self.loss_weight_velocity * ((v - v_pred) ** 2).mean()

            if self.loss_weight_stft > 0:
                fft_sizes = [64, 128, 256, 512, 1024]
                hop_sizes = [16, 32, 64, 128, 256]
                win_lengths = [64, 128, 256, 512, 1024]
                signal_length = x.reshape(x.shape[0], x.shape[1], -1).shape[-1]
                stft_configs = [
                    (n_fft, hop, win)
                    for n_fft, hop, win in zip(fft_sizes, hop_sizes, win_lengths)
                    if n_fft <= signal_length
                ]

                loss_stft = x.new_tensor(0.0)
                for n_fft, hop, win in stft_configs:
                    loss_stft += self._stft_loss_multires(x, x_pred, n_fft, hop, win)
                if stft_configs:
                    loss_stft /= len(stft_configs)
                    loss = loss + self.loss_weight_stft * loss_stft

            if self.loss_weight_stat > 0:
                loss = loss + self.loss_weight_stat * self._statistics_loss(x, x_pred)

            if self.loss_weight_tv > 0:
                loss = loss + self.loss_weight_tv * self._total_variation_loss(x_pred)

            if self.loss_weight_corr > 0:
                loss = loss + self.loss_weight_corr * self._pearson_corr_loss(x, x_pred)

            return loss

        if self.loss_type == "l1":
            return (v - v_pred).abs().mean()
        return ((v - v_pred) ** 2).mean()

    @staticmethod
    def _stft_loss_multires(x, x_pred, n_fft, hop_length, win_length):
        window = torch.hann_window(win_length).to(x.device)

        def get_magnitude(tensor):
            b, c = tensor.shape[:2]
            tensor = tensor.view(b * c, -1).float()
            stft = torch.stft(tensor, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)
            return torch.abs(stft)

        mag_x = get_magnitude(x)
        mag_pred = get_magnitude(x_pred)

        sc_loss = torch.norm(mag_x - mag_pred, p="fro") / (torch.norm(mag_x, p="fro") + 1e-7)
        lm_loss = F.l1_loss(torch.log(mag_x + 1e-7), torch.log(mag_pred + 1e-7))
        return sc_loss + lm_loss

    @staticmethod
    def _statistics_loss(x, x_pred):
        std_x = x.std(dim=-1)
        std_pred = x_pred.std(dim=-1)
        mean_x = x.mean(dim=-1)
        mean_pred = x_pred.mean(dim=-1)
        return F.l1_loss(std_pred, std_x) + F.l1_loss(mean_pred, mean_x)

    @staticmethod
    def _total_variation_loss(x):
        return (x[..., 1:] - x[..., :-1]).abs().mean()

    @staticmethod
    def _pearson_corr_loss(x, x_pred):
        x_f = x.reshape(x.shape[0], -1)
        pred_f = x_pred.reshape(x_pred.shape[0], -1)
        x_center = x_f - x_f.mean(dim=1, keepdim=True)
        pred_center = pred_f - pred_f.mean(dim=1, keepdim=True)
        numerator = (x_center * pred_center).sum(dim=1)
        denominator = torch.sqrt((x_center.pow(2).sum(dim=1) + 1e-8) * (pred_center.pow(2).sum(dim=1) + 1e-8))
        corr = numerator / denominator
        return (1.0 - corr.clamp(-1.0, 1.0)).mean()

    @torch.no_grad()
    def update_ema(self):
        for p, p_ema1, p_ema2 in zip(self.parameters(), self.ema_params1, self.ema_params2):
            p_ema1.mul_(self.ema_decay1).add_(p.data, alpha=1.0 - self.ema_decay1)
            p_ema2.mul_(self.ema_decay2).add_(p.data, alpha=1.0 - self.ema_decay2)

    @torch.no_grad()
    def _run_net(self, z, t, use_ema=True):
        if use_ema is False or use_ema == "raw":
            return self.net(z, t)

        backup = [p.detach().clone() for p in self.parameters()]
        if use_ema == "ema2":
            ema = self.ema_params2 if self.ema_params2 is not None else backup
        else:
            ema = self.ema_params1 if self.ema_params1 is not None else backup
        for p, p_ema in zip(self.parameters(), ema):
            p.data.copy_(p_ema)

        out = self.net(z, t)

        for p, old in zip(self.parameters(), backup):
            p.data.copy_(old)
        return out

    @torch.no_grad()
    def _run_clean_pred(self, z, t, use_ema=True):
        if use_ema is False or use_ema == "raw":
            return self._clean_pred_from_net(z, t)

        backup = [p.detach().clone() for p in self.parameters()]
        if use_ema == "ema2":
            ema = self.ema_params2 if self.ema_params2 is not None else backup
        else:
            ema = self.ema_params1 if self.ema_params1 is not None else backup
        for p, p_ema in zip(self.parameters(), ema):
            p.data.copy_(p_ema)

        out = self._clean_pred_from_net(z, t)

        for p, old in zip(self.parameters(), backup):
            p.data.copy_(old)
        return out

    @torch.no_grad()
    def denoise(self, noisy, use_ema=True):
        z = self.to_training_space(noisy)
        b = z.shape[0]
        device = z.device

        if self.prediction_target == "clean" and self.denoise_mode == "direct":
            t_vec = torch.zeros((b,), device=device)
            return self._run_clean_pred(z, t_vec, use_ema=use_ema)

        steps = self.steps
        use_heun = self.method == "heun"
        t_schedule = torch.linspace(0.0, 1.0, steps + 1, device=device)

        def _get_v(z_in, t_scalar):
            t_vec = torch.full((b,), t_scalar, device=device)
            if self.prediction_target == "clean":
                clean_pred = self._run_clean_pred(z_in, t_vec, use_ema=use_ema)
                denom = max(1.0 - t_scalar, self.t_eps)
                return (clean_pred - z_in) / denom
            net_out = self._run_net(z_in, t_vec, use_ema=use_ema)
            return net_out

        for i in range(steps):
            t_cur = float(t_schedule[i])
            t_nxt = float(t_schedule[i + 1])
            dt = t_nxt - t_cur

            v1 = _get_v(z, t_cur)

            if use_heun and i < steps - 1:
                z_euler = z + dt * v1
                v2 = _get_v(z_euler, t_nxt)
                z = z + dt * 0.5 * (v1 + v2)
            else:
                z = z + dt * v1

        return z

    @torch.no_grad()
    def generate(self, noisy, **_kwargs):
        return self.denoise(noisy)
