# Copyright (c) OpenMMLab. All rights reserved.

from typing import Union

import torch

from mmrazor.registry import MODELS
from ..derived_mutable import DerivedMutable
from .base_mutable_channel import BaseMutableChannel


@MODELS.register_module()
class SimpleMutableChannel(BaseMutableChannel):
    """SimpleMutableChannel is a simple BaseMutableChannel, it directly take a
    mask as a choice.

    Args:
        num_channels (int): number of channels.
    """

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        mask = torch.ones([self.num_channels
                           ])  # save bool as float for dist training
        self.register_buffer('mask', mask)
        self.mask: torch.Tensor

    # choice

    @property
    def current_choice(self) -> torch.Tensor:
        """Get current choice."""
        return self.mask.bool()

    @current_choice.setter
    def current_choice(self, choice: torch.Tensor):
        """Set current choice."""
        self.mask = choice.to(self.mask.device).float()

    @property
    def current_mask(self) -> torch.Tensor:
        """Get current mask."""
        return self.current_choice.bool()

    # basic extension

    def expand_mutable_channel(
            self, expand_ratio: Union[int, float]) -> DerivedMutable:
        """Get a derived SimpleMutableChannel with expanded mask."""

        def _expand_mask():
            mask = self.current_mask
            mask = torch.unsqueeze(
                mask, -1).expand(list(mask.shape) + [expand_ratio]).flatten(-2)
            return mask

        return DerivedMutable(_expand_mask, _expand_mask, [self])
