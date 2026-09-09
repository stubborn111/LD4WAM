"""Action backbone package: the action-stream ABCs + concrete implementations.

    from openwam.model.action_backbone import ActionDiT

There is no registry — each architecture constructs its action backbone
directly (la_tri_system builds ``ActionDiT`` with the ``idm`` variant), so this
file only re-exports the public classes.
"""

from openwam.model.action_backbone.base import (
    ActionDiTBackbone,
    SharedActionBackbone,
)
from openwam.model.action_backbone.scheduler import ActionScheduler
from openwam.model.action_backbone.separate_action_dit import ActionDiT

__all__ = [
    "ActionDiT",
    "ActionScheduler",
    "ActionDiTBackbone",
    "SharedActionBackbone",
]
