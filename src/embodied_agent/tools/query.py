"""Read-only model tools over a captured world; never step or save a simulator."""
from __future__ import annotations

import copy
from typing import Any

from embodied_agent.maps.home_schema import HomeWorld
from embodied_agent.models.contracts import PlannerError
from embodied_agent.tools.capabilities import build_capabilities
from embodied_agent.agents.observation import WorldObservation
from embodied_agent.tools.registry import QueryToolCatalog

class QueryTools:
    """Expose structured state and capabilities without interpreting instructions.

    The snapshot is captured before the language worker starts. No model tool
    touches MuJoCo, persistent maps, policies or execution controls.
    """

    def __init__(self, world: Any, snapshot: dict | None = None, *, edit_mode: bool = False,
                 capabilities: dict | None = None, catalog: QueryToolCatalog | None = None):
        self.world, self.edit_mode = world, edit_mode
        if snapshot is None:
            snapshot = world.snapshot() if isinstance(world, HomeWorld) else world.to_dict()
        self.snapshot = copy.deepcopy(snapshot)
        self.snapshot.pop("untrusted_descriptions", None)
        self.snapshot.pop("events", None)
        self._view = WorldObservation.capture(self.snapshot, world)
        self._capabilities = copy.deepcopy(capabilities)
        self.catalog = catalog if catalog is not None else QueryToolCatalog()

    @property
    def schemas(self) -> list[dict]:
        return self.catalog.schemas

    def call(self, name: str, arguments: dict) -> dict:
        return self.catalog.dispatch(self, name, arguments)

    def _observe(self) -> dict:
        return {"source": "captured_declared_simulation_state", "snapshot": copy.deepcopy(self.snapshot)}

    def _query_entity(self, object_id: str, *, inspect: bool) -> dict:
        if object_id in self._view.objects:
            value = copy.deepcopy(self._view.objects[object_id])
            if inspect and "inspect" not in value.get("operations", []):
                raise PlannerError("UNSUPPORTED_OPERATION", f"Inspection is not permitted for {object_id}")
        elif not inspect and object_id in self._view.supports:
            value = copy.deepcopy(self._view.supports[object_id])
        elif not inspect and object_id == "danger_zone" and self._view.domain == "desktop":
            value = self.world.to_dict()["danger_zone"]
        else:
            raise PlannerError("UNKNOWN_OBJECT", f"No queryable entity {object_id!r}")
        return {"source": "captured_declared_simulation_state", "object_id": object_id,
                "object": copy.deepcopy(value), "world_version": self.snapshot.get("world_version"),
                "map_revision": self.world.revision}

    def capabilities(self) -> dict:
        if self.edit_mode:
            return {"domain": "home" if isinstance(self.world, HomeWorld) else "desktop",
                "kind": "initial_map_edits", "base_revision": self.world.revision,
                "operations": (["set_position", "set_name", "set_description"] if isinstance(self.world, HomeWorld)
                               else ["set_position", "set_half_size", "scale", "shift_axis"]),
                "object_ids": (list(self.world.objects) if isinstance(self.world, HomeWorld)
                               else ["cube", "danger_zone", *self.world.targets]),
                "constraints": "Edits are validated atomically; tools never save. Home furniture, permissions, risk states and policy are not editable."}
        if self._capabilities is None:
            self._capabilities = build_capabilities(self.world, self.snapshot)
        capabilities = copy.deepcopy(self._capabilities)
        capabilities["query_tools"] = list(self.catalog.names)
        return capabilities
