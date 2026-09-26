import math
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from util import misc


class PersistentIREPA(nn.Module):
    def __init__(
        self,
        feature_dim,
        *,
        weight=0.1,
        earlystop=100,
        kernel_size=3,
        spnorm_alpha=0.8,
        spnorm_eps=1e-6,
        teacher_repo="facebookresearch/dinov2",
        teacher_source="github",
        teacher_name="dinov2_vitb14",
        teacher_checkpoint="",
        teacher_dim=768,
        teacher_input_size=224,
    ):
        super().__init__()
        if earlystop < 0 or weight < 0 or kernel_size <= 0 or kernel_size % 2 != 1:
            raise ValueError("Invalid iREPA earlystop, weight or kernel size")
        self.weight, self.earlystop = weight, earlystop
        self.active = weight > 0 and earlystop > 0
        self.feature_dim = feature_dim
        self.spnorm_alpha, self.spnorm_eps = spnorm_alpha, spnorm_eps
        self.dino_feature_dim, self.dino_input_size = teacher_dim, teacher_input_size
        self.teacher_repo, self.teacher_source = teacher_repo, teacher_source
        self.teacher_name, self.teacher_checkpoint = teacher_name, teacher_checkpoint
        self.projector = nn.Conv2d(
            feature_dim, teacher_dim, kernel_size, padding=kernel_size // 2
        )
        object.__setattr__(self, "_dino_teacher", None)
        object.__setattr__(self, "_dino_teacher_device", None)
        self.register_buffer(
            "_dino_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_dino_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def set_epoch(self, epoch):
        self.active = self.weight > 0 and epoch < self.earlystop
        if not self.active:
            object.__setattr__(self, "_dino_teacher", None)
            object.__setattr__(self, "_dino_teacher_device", None)
        return self.active

    def _ensure_dino_teacher_device(self, device):
        if self._dino_teacher is None:
            # Lazy teacher construction must not advance the training RNG on resume.
            rng = misc.rng_state(device)
            try:
                # Serialize hub initialization on each shared cache (multi-process training).
                import fcntl

                hub = Path(torch.hub.get_dir())
                hub.mkdir(parents=True, exist_ok=True)
                with (hub / ".perf_dinov2.lock").open("w") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    teacher = torch.hub.load(
                        self.teacher_repo,
                        self.teacher_name,
                        source=self.teacher_source,
                        trust_repo=True,
                        pretrained=not bool(self.teacher_checkpoint),
                    )
                    fcntl.flock(lock, fcntl.LOCK_UN)
                if self.teacher_checkpoint:
                    checkpoint = torch.load(
                        self.teacher_checkpoint, map_location="cpu", weights_only=True
                    )
                    teacher.load_state_dict(
                        checkpoint.get("model", checkpoint), strict=True
                    )
                teacher.eval().requires_grad_(False)
                object.__setattr__(self, "_dino_teacher", teacher)
            finally:
                misc.restore_rng(rng, device)
        if self._dino_teacher_device != device:
            self._dino_teacher.to(device)
            object.__setattr__(self, "_dino_teacher_device", device)
        self._dino_teacher.eval()

    def _preprocess_dino_images(self, images):
        images = ((images.float() + 1.0) * 0.5).clamp(0.0, 1.0)
        mean = self._dino_mean.to(device=images.device, dtype=images.dtype)
        std = self._dino_std.to(device=images.device, dtype=images.dtype)
        images = (images - mean) / std
        return F.interpolate(
            images,
            size=(self.dino_input_size, self.dino_input_size),
            mode="bicubic",
            align_corners=False,
        )

    def _spatial_zscore(self, features):
        mean = features.mean(dim=1, keepdim=True)
        std = features.std(dim=1, keepdim=True)
        return (features - self.spnorm_alpha * mean) / (std + self.spnorm_eps)

    @torch.no_grad()
    def _forward_dino_teacher(self, images):
        self._ensure_dino_teacher_device(images.device)
        dino_images = self._preprocess_dino_images(images)
        autocast_context = (
            torch.amp.autocast("cuda", enabled=False)
            if dino_images.device.type == "cuda"
            else nullcontext()
        )
        with autocast_context:
            features = self._dino_teacher.forward_features(dino_images.float())

        if not isinstance(features, dict) or "x_norm_patchtokens" not in features:
            keys = (
                list(features.keys())
                if isinstance(features, dict)
                else type(features).__name__
            )
            raise KeyError(f"Expected DINOv2 x_norm_patchtokens, got {keys}.")
        patch_tokens = features["x_norm_patchtokens"]
        expected_tokens = (self.dino_input_size // 14) ** 2
        if patch_tokens.shape != (
            images.shape[0],
            expected_tokens,
            self.dino_feature_dim,
        ):
            raise ValueError(
                "Unexpected DINOv2 patch-token shape: "
                f"expected {(images.shape[0], expected_tokens, self.dino_feature_dim)}, "
                f"got {tuple(patch_tokens.shape)}."
            )
        return self._spatial_zscore(patch_tokens.float())

    def _project_persistent_feature(self, persistent_feature):
        batch_size, num_tokens, feature_dim = persistent_feature.shape
        grid_size = math.isqrt(num_tokens)
        if grid_size * grid_size != num_tokens:
            raise ValueError(
                "The persistent iREPA Conv2d projector requires a square token grid, "
                f"got {num_tokens} tokens."
            )
        if feature_dim != self.feature_dim:
            raise ValueError(
                f"Expected persistent feature dim {self.feature_dim}, "
                f"got {feature_dim}."
            )

        feature_map = persistent_feature.reshape(
            batch_size, grid_size, grid_size, feature_dim
        )
        feature_map = feature_map.permute(0, 3, 1, 2).contiguous()
        projected = self.projector(feature_map)
        return projected.flatten(2).transpose(1, 2).contiguous()

    def _zero_projector_graph_term(self):
        # DDP was constructed while the projector required gradients. Keep a
        # zero-valued graph edge after early stop so find_unused_parameters is
        # not required and the optimizer parameter layout stays unchanged.
        zero = None
        for parameter in self.projector.parameters():
            term = parameter.reshape(-1)[0] * 0.0
            zero = term if zero is None else zero + term
        if zero is None:
            raise RuntimeError("iREPA projector unexpectedly has no parameters.")
        return zero

    def forward(self, features, images):
        student = F.normalize(
            self._project_persistent_feature(features).float(), dim=-1
        )
        teacher = F.normalize(self._forward_dino_teacher(images), dim=-1)
        if student.shape != teacher.shape:
            raise ValueError(
                f"iREPA feature mismatch: {student.shape} vs {teacher.shape}"
            )
        return 1.0 - (student * teacher).sum(dim=-1).mean()

    def inactive_loss(self):
        # Keep zero gradients for DDP while retaining a stable optimizer layout.
        return self._zero_projector_graph_term()
