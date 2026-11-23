"""LeJEPA model with its loss and optional encoder features."""

import torch
from torch import nn
from torchvision.ops import MLP


class LeJEPA(nn.Module):
    """Pull projected views toward their mean and regularize them with SIGReg.

    The vision encoder must return one feature vector per image and expose
    its output dimension through ``num_features``, as timm encoders do.

    Pass a configured SIGReg module that accepts projections shaped
    ``(num_views, batch_size, feature_dim)`` and returns a scalar loss
    averaged across views. Its buffers are registered as part of this model.

    ``proj_dim`` is the projector's output dimension, where the loss is
    computed.
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        sigreg: nn.Module,
        lamb: float = 0.01,
        proj_dim: int = 512,
    ):
        super().__init__()
        self.encoder = vision_encoder
        self.lamb = lamb

        embed_dim = vision_encoder.num_features
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, 512, bias=True),
            MLP(
                in_channels=512,
                hidden_channels=[2048, 2048, proj_dim],
                norm_layer=nn.SyncBatchNorm,
            ),
        )
        self.sigreg = sigreg

    def forward(self, views: torch.Tensor, get_features: bool = False):
        """Return a loss dictionary for views shaped ``(B, V, C, H, W)``.

        The dictionary contains the weighted ``ssl_loss`` and its components,
        ``mse_loss`` (the mean squared distance of each view's projection to
        the mean over views) and ``sigreg_loss``, before weighting by
        ``lamb``. All losses retain gradients.

        With ``get_features=True``, return ``(losses, features)``. Features
        have shape ``(B * V, D)``, with each image's views adjacent.
        They retain gradients; detach them when training an online probe
        so the probe loss does not update the encoder.
        """
        batch_size, num_views = views.shape[:2]
        features = self.encoder(views.flatten(0, 1))
        projections = self.projector(features)
        projections = projections.reshape(batch_size, num_views, -1)

        # Keep the loss in FP32 even when Fabric autocasts the encoder.
        with torch.autocast(
            device_type=projections.device.type, enabled=False
        ):
            projections = projections.float()
            # Pull every view toward the mean over views (LeJEPA's center).
            centers = projections.mean(dim=1, keepdim=True)
            mse_loss = (projections - centers).square().mean()

            # SIGReg uses (views, batch, features) to keep views separate.
            sigreg_loss = self.sigreg(projections.transpose(0, 1))
            loss = (1 - self.lamb) * mse_loss + self.lamb * sigreg_loss

        losses = {
            "ssl_loss": loss,
            "mse_loss": mse_loss,
            "sigreg_loss": sigreg_loss,
        }
        if get_features:
            return losses, features
        return losses
