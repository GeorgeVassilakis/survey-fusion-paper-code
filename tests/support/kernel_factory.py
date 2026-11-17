from __future__ import annotations

import torch

from survey_fusion.pgm.contracts import GroupedKernelSamples


class PlanPairKernelFactoryMixin:
    def build_kernel_sample_pair_from_plan(
        self, plan: torch.Tensor, *, k: int, device: torch.device,
    ) -> tuple[GroupedKernelSamples, GroupedKernelSamples]:
        return (
            self.build_kernel_samples_from_plan(plan, k=k, device=device),
            self.build_kernel_samples_from_plan(plan, k=k, device=device),
        )
