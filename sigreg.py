"""Gaussian regularization for batches of projected representations."""

# Adapted from the official LeJEPA implementation
# (https://github.com/galilai-group/lejepa), licensed under CC BY-NC 4.0.

import torch
import torch.distributed as dist
from torch.distributed.nn import all_reduce as functional_all_reduce


class SIGReg(torch.nn.Module):
    """Sketched isotropic Gaussian regularizer.

    Regularizes a marginal embedding distribution toward an isotropic Gaussian
    via an Epps-Pulley goodness-of-fit statistic on random 1-D projections.

    With ``distributed=True`` and an initialized process group, all ranks use
    rank zero's projection directions and differentiably average their
    empirical characteristic functions. Distributed mode assumes equal local
    batch sizes and scales the statistic by local batch size times world size.
    Without distributed aggregation, it uses the local batch size.
    """

    def __init__(self, knots=17, num_proj=4096, distributed=False):
        super().__init__()
        self.num_proj = num_proj
        self.distributed = distributed
        integration_grid = torch.linspace(0, 3, knots, dtype=torch.float32)
        step_size = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * step_size, dtype=torch.float32)
        weights[[0, -1]] = step_size
        window = torch.exp(-integration_grid.square() / 2.0)
        self.register_buffer("t", integration_grid)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """Regularize projected representations.

        ``proj`` has shape ``(..., batch_size, feature_dim)``.
        """
        use_distributed = (
            self.distributed
            and dist.is_available()
            and dist.is_initialized()
        )
        world_size = dist.get_world_size() if use_distributed else 1

        # Sample unit-length directions for the random projections.
        directions = torch.randn(
            proj.size(-1), self.num_proj, device=proj.device
        )
        directions = directions.div_(directions.norm(p=2, dim=0))
        if use_distributed:
            # Use the same projections when pooling across ranks.
            dist.broadcast(directions, src=0)

        # Evaluate projected samples on the integration grid.
        projected_samples = (proj @ directions).unsqueeze(-1) * self.t

        if use_distributed:
            # Average across ranks with gradients through the collective.
            empirical_cf = torch.stack(
                (
                    projected_samples.cos().mean(-3),
                    projected_samples.sin().mean(-3),
                )
            )
            empirical_cf = functional_all_reduce(
                empirical_cf, op=dist.ReduceOp.SUM
            ) / world_size
            squared_error = (
                (empirical_cf[0] - self.phi).square()
                + empirical_cf[1].square()
            )
        else:
            squared_error = (
                (projected_samples.cos().mean(-3) - self.phi).square()
                + projected_samples.sin().mean(-3).square()
            )

        # Scale by the sample count used to estimate the distribution.
        batch_size = proj.size(-2) * world_size
        statistic = (squared_error @ self.weights) * batch_size
        return statistic.mean()
