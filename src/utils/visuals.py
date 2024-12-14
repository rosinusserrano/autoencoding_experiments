"""Functions for plotting."""

import torch
from torchvision.utils import make_grid


def show_side_by_side(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
) -> torch.Tensor:
    """Take to batches of images and show them side by side."""
    row1 = make_grid(
        tensor1,
        normalize=True,
        padding=1,
        pad_value=1,
    )

    row2 = make_grid(
        tensor2,
        normalize=True,
        padding=1,
        pad_value=1,
    )

    return make_grid(
        torch.cat((row1, row2), dim=1),
        nrow=1,
        padding=1,
        pad_value=1,
    )
