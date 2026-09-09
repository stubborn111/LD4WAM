"""Side-effect import of the la_tri_system framework architectures.

``la_tri_system`` = latent-action tri-system (video + latent_action + action).
Currently one variant: ``idm`` (FastWAM-IDM two-stage). The framework name is
structure-agnostic so future non-IDM variants register here too.
"""

from openwam.model.architectures.la_tri_system.idm import LaTriSystemIDMArchitecture
from openwam.model.architectures.la_tri_system.mot_driver import LaTriSystemIDMMoTDriver

__all__ = ["LaTriSystemIDMArchitecture", "LaTriSystemIDMMoTDriver"]
