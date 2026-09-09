"""Online Latent Action Model (LAM) port for the ``la_tri_system`` framework.

Public surface: the :class:`LatentActionModel` interface and the
:func:`build_latent_action_model` factory. Concrete adapters self-register on
import (LAQ adapter below); OpenWAM never imports a LAM implementation directly.
"""

# Side-effect import so the built-in adapters register. The adapter itself does
# NOT import the LAM implementation at module load — that happens lazily in
# ``from_config`` via a dynamic import from the configured ``repo_path``.
from openwam.model.latent_action_model import laq_adapter  # noqa: F401,E402
from openwam.model.latent_action_model.base import LatentActionModel
from openwam.model.latent_action_model.registry import (
    build_latent_action_model,
    register_latent_action_model,
)

__all__ = [
    "LatentActionModel",
    "build_latent_action_model",
    "register_latent_action_model",
]
