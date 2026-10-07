"""One registry supplies the model schemas and read-only query dispatch."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Mapping

from embodied_agent.models.contracts import PlannerError


@dataclass(frozen=True)
class QueryToolDefinition:
    """A trusted implementation and its closed input schema, versioned together.

    Replacement definitions can change descriptions, narrow allowed IDs, or use
    another read-only handler. They cannot register new model permissions or
    change the existing argument names/types. Handlers receive a QueryTools
    context and detached, validated arguments.
    """

    name: str
    description: str
    parameters: dict
    handler: Callable[[Any, dict], dict]
    version: str = "1"

    @property
    def schema(self) -> dict:
        return {"type": "function", "function": {"name": self.name,
            "description": self.description, "parameters": copy.deepcopy(self.parameters)}}


def _parameters(*names: str) -> dict:
    return {"type": "object", "properties": {name: {"type": "string"} for name in names},
            "required": list(names), "additionalProperties": False}


def _observe(context, arguments):
    return context._observe()


def _query_world(context, arguments):
    return context._query_entity(arguments["object_id"], inspect=False)


def _inspect(context, arguments):
    return context._query_entity(arguments["object_id"], inspect=True)


def _get_capabilities(context, arguments):
    return context.capabilities()


_DEFAULTS = {
    definition.name: definition for definition in (
        QueryToolDefinition("observe", "Read the captured current world state.", _parameters(), _observe),
        QueryToolDefinition("query_world", "Read one registered object's or support's current state.",
                            _parameters("object_id"), _query_world),
        QueryToolDefinition("inspect", "Read one registered object's state and inspection permission.",
                            _parameters("object_id"), _inspect),
        QueryToolDefinition("get_capabilities", "Read available actions or edits, verified operation points and constraints.",
                            _parameters(), _get_capabilities),
    )
}
QUERY_NAMES = tuple(_DEFAULTS)


def _configuration_error(message: str) -> None:
    raise PlannerError("INVALID_TOOL_CONFIG", message)


def _validate_definition(definition: QueryToolDefinition) -> None:
    if (not isinstance(definition, QueryToolDefinition) or not isinstance(definition.name, str)
            or definition.name not in _DEFAULTS):
        _configuration_error("Only the registered read-only query tools may be replaced")
    if (not isinstance(definition.description, str) or not definition.description.strip()
            or not isinstance(definition.version, str) or not definition.version.strip()
            or not callable(definition.handler)):
        _configuration_error("Tool definitions require a description, version and callable handler")
    parameters = definition.parameters
    baseline = _DEFAULTS[definition.name].parameters
    if (not isinstance(parameters, dict)
            or set(parameters) != {"type", "properties", "required", "additionalProperties"}
            or parameters["type"] != "object" or parameters["additionalProperties"] is not False
            or not isinstance(parameters["properties"], dict)
            or set(parameters["properties"]) != set(baseline["properties"])
            or parameters["required"] != baseline["required"]):
        _configuration_error(f"Replacement for {definition.name} must retain the closed argument contract")
    for name, schema in parameters["properties"].items():
        if (not isinstance(schema, dict) or schema.get("type") != "string"
                or set(schema) - {"type", "description", "enum", "minLength", "maxLength"}):
            _configuration_error(f"Invalid string schema for {definition.name}.{name}")
        if "description" in schema and not isinstance(schema["description"], str):
            _configuration_error("Argument descriptions must be strings")
        if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]
                or any(not isinstance(value, str) or not value for value in schema["enum"])):
            _configuration_error("ID enums require a nonempty list of nonempty strings")
        for bound in ("minLength", "maxLength"):
            if bound in schema and (type(schema[bound]) is not int or schema[bound] < 0):
                _configuration_error("String length bounds must be nonnegative integers")
        if schema.get("minLength", 0) > schema.get("maxLength", float("inf")):
            _configuration_error("String length bounds are inconsistent")


class QueryToolCatalog:
    """Build an independent candidate from components.tools in a runtime dict.

    query_names selects an ordered subset (including an empty subset).
    descriptions overrides published tool descriptions. definitions supplies
    replacement QueryToolDefinition objects for existing tool names; all schema
    validation and dispatch use the same detached definitions.
    """

    version = "query-tools-v1"

    def __init__(self, config: dict | None = None, *,
                 definitions: Mapping[str, QueryToolDefinition] | None = None):
        config = {} if config is None else config
        if not isinstance(config, dict) or not isinstance(config.get("components", {}), dict):
            _configuration_error("Tool configuration requires a runtime dictionary with components")
        settings = config.get("components", {}).get("tools", {})
        if not isinstance(settings, dict) or set(settings) - {"query_names", "descriptions"}:
            _configuration_error("components.tools supports query_names and descriptions")
        names = settings.get("query_names", list(QUERY_NAMES))
        if (not isinstance(names, list) or any(not isinstance(name, str) or name not in _DEFAULTS for name in names)
                or len(set(names)) != len(names)):
            _configuration_error("query_names requires a unique subset of registered query names")
        descriptions = settings.get("descriptions", {})
        if (not isinstance(descriptions, dict) or any(name not in _DEFAULTS for name in descriptions)
                or any(not isinstance(value, str) or not value.strip() for value in descriptions.values())):
            _configuration_error("descriptions requires registered names and nonempty strings")
        candidates = dict(_DEFAULTS)
        if definitions is not None:
            if not isinstance(definitions, Mapping):
                _configuration_error("definitions must map registered names to QueryToolDefinition objects")
            for name, definition in definitions.items():
                _validate_definition(definition)
                if name != definition.name:
                    _configuration_error("Definition registry keys must equal their tool names")
                candidates[name] = definition
        self._definitions = {}
        for name in names:
            definition = candidates[name]
            _validate_definition(definition)
            self._definitions[name] = QueryToolDefinition(name,
                descriptions.get(name, definition.description), copy.deepcopy(definition.parameters),
                definition.handler, definition.version)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._definitions)

    @property
    def schemas(self) -> list[dict]:
        return [definition.schema for definition in self._definitions.values()]

    @property
    def fingerprint(self) -> str:
        definition_ids = [{"schema": definition.schema, "version": definition.version,
            "handler": f"{definition.handler.__module__}.{getattr(definition.handler, '__qualname__', type(definition.handler).__qualname__)}"}
            for definition in self._definitions.values()]
        encoded = json.dumps({"version": self.version, "definitions": definition_ids},
            sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def dispatch(self, context: Any, name: str, arguments: dict) -> dict:
        if not isinstance(name, str) or name not in self._definitions:
            raise PlannerError("UNKNOWN_TOOL", f"Query tool {name!r} is not registered or is disabled")
        definition = self._definitions[name]
        parameters = definition.parameters
        if not isinstance(arguments, dict) or set(arguments) != set(parameters["required"]):
            raise PlannerError("INVALID_TOOL_ARGUMENTS", f"Unexpected arguments for {name}")
        for field, schema in parameters["properties"].items():
            value = arguments[field]
            if (not isinstance(value, str) or not value
                    or ("enum" in schema and value not in schema["enum"])
                    or len(value) < schema.get("minLength", 0)
                    or len(value) > schema.get("maxLength", float("inf"))):
                raise PlannerError("INVALID_TOOL_ARGUMENTS", f"{field} requires a permitted registered ID")
        return definition.handler(context, copy.deepcopy(arguments))
