"""tajfel — Tajfel (1971) minimal-group paradigm. Importing the package registers its four abc
implementations (env/pop/decision/collector) so engine.build_simulator can resolve them."""

from __future__ import annotations

from . import model  # noqa: F401  (side effect: @register decorators run)
from .model import make_tajfel_bundles  # noqa: F401

__all__ = ["make_tajfel_bundles", "model"]
