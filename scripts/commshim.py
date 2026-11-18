"""Local end-to-end check of the E-series CLI without the SocioVerse runtime.

`scripts/run_communication.py` imports the real `socioverse` package and `tajfel.model`. Neither
is installed on this Windows interpreter, so this module installs a minimal stand-in for exactly
the names those imports touch, then runs the *same* CLI code paths that WSL will run:

  python scripts/run_communication.py --condition all --llm scripted --driver standalone

The stand-in only provides registration plumbing and schema containers. Every decision, routing
rule, permission check and export exercised here is the real implementation in
`tajfel/communication.py`.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class _Schema:
    """Permissive pydantic-lite stand-in: keeps kwargs, tolerates attribute access."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def model_copy(self, *, update=None, deep=False):
        clone = _Schema(**{k: v for k, v in vars(self).items()})
        for k, v in (update or {}).items():
            setattr(clone, k, v)
        return clone

    def model_dump(self, **kwargs):
        return dict(vars(self))

    def __repr__(self):
        return f"{type(self).__name__}({vars(self)!r})"


REGISTRY: dict[tuple[str, str], type] = {}


def _register(kind, key):
    def deco(cls):
        REGISTRY[(kind, key)] = cls
        return cls
    return deco


class _ProviderBase:
    def __init__(self, *a, **kw):
        pass


def install() -> dict:
    """Install the stub modules. Must run before `import tajfel`."""
    if "socioverse" in sys.modules:
        return REGISTRY

    for name in ("socioverse", "socioverse.abc", "socioverse.engine", "socioverse.schemas"):
        sys.modules[name] = types.ModuleType(name)

    for name in ("DecisionModel", "EnvironmentProvider", "MetricCollector", "PopulationProvider"):
        setattr(sys.modules["socioverse.abc"], name, _ProviderBase)
    sys.modules["socioverse.engine"].register = _register

    # every schema class model.py / communication.py import, plus the bundle containers
    schemas = sys.modules["socioverse.schemas"]
    for name in ("Action", "Observation", "Broadcast", "EnvironmentBundle", "EnvironmentLayer",
                 "InformationProgram", "InteractionStructure", "Persona", "PopulationBundle",
                 "PropagationMode", "ScheduledEvent", "SimulationConfig", "StudySpec"):
        setattr(schemas, name, type(name, (_Schema,), {}))

    return REGISTRY
