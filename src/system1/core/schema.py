"""System 1 Core Decision Schema Definitions.

Provides strongly-typed, machine-native schema fields for non-autoregressive
decision evaluation:
- DecisionField: Abstract base class for schema field descriptors.
- ChoiceField: Discrete multi-class selection with calibrated probabilities.
- BooleanField: Binary decision (allow/deny, safe/unsafe, triage flag).
- MultiChoiceField: Multi-label classification with independent probabilities.
- ScoreField: Bounded continuous regression score in [min_value, max_value].
- DecisionSchema: Declarative schema container with canonical SHA-256 fingerprinting.

Zero external dependencies: Requires only standard library (hashlib, json, re, abc, dataclasses).
"""

from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Mapping, Optional, Sequence, Tuple, Type, Union, get_type_hints


_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")


def _validate_field_name(name: str) -> str:
    if not isinstance(name, str) or not _IDENTIFIER_PATTERN.fullmatch(name):
        raise ValueError(
            f"Invalid field name {name!r}: must match pattern ^[A-Za-z_][A-Za-z0-9_]{{0,127}}$"
        )
    return name


class DecisionField(ABC):
    """Abstract base class for typed Reflex decision schema fields."""

    field_type: ClassVar[str]

    def __init__(
        self,
        *,
        description: str = "",
        required: bool = True,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.name: str = ""
        self.description: str = str(description).strip()
        self.required: bool = bool(required)
        self.metadata: Dict[str, Any] = dict(metadata or {})

    def bind_name(self, name: str) -> None:
        """Binds the field name once declared in a schema."""
        self.name = _validate_field_name(name)

    @abstractmethod
    def validate_value(self, value: Any) -> Any:
        """Validates and coerces an evaluated value for this field."""
        raise NotImplementedError

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serializes the field specification to a canonical dictionary."""
        raise NotImplementedError

    def fingerprint(self) -> str:
        """Returns the canonical SHA-256 digest of this field definition."""
        canonical_json = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


class ChoiceField(DecisionField):
    """Discrete categorical choice field among a fixed set of options."""

    field_type: ClassVar[str] = "choice"

    def __init__(
        self,
        options: Sequence[str],
        *,
        descriptions: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
        default: Optional[str] = None,
        description: str = "",
        required: bool = True,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(description=description, required=required, metadata=metadata)
        if not options:
            raise ValueError("ChoiceField requires at least one option")
        cleaned_options = []
        seen = set()
        for opt in options:
            if not isinstance(opt, str) or not opt.strip():
                raise ValueError(f"ChoiceField options must be non-empty strings, got {opt!r}")
            cleaned = opt.strip()
            if cleaned in seen:
                raise ValueError(f"Duplicate option in ChoiceField: {cleaned!r}")
            seen.add(cleaned)
            cleaned_options.append(cleaned)
        self.options: Tuple[str, ...] = tuple(cleaned_options)
        parsed_descriptions: Dict[str, Union[str, Tuple[str, ...]]] = {}
        for k, v in (descriptions or {}).items():
            if k in seen:
                if isinstance(v, str):
                    parsed_descriptions[k] = v.strip()
                elif isinstance(v, (list, tuple, set)):
                    parsed_descriptions[k] = tuple(str(x).strip() for x in v if str(x).strip())
                else:
                    parsed_descriptions[k] = str(v).strip()
        self.descriptions: Dict[str, Union[str, Tuple[str, ...]]] = parsed_descriptions
        if default is not None:
            if default not in seen:
                raise ValueError(f"Default value {default!r} not in options {self.options}")
        self.default: Optional[str] = default

    def validate_value(self, value: Any) -> str:
        if value is None and self.default is not None:
            return self.default
        if not isinstance(value, str):
            raise TypeError(f"ChoiceField {self.name!r} expected str, got {type(value).__name__}")
        if value not in self.options:
            raise ValueError(
                f"Value {value!r} not permitted for field {self.name!r}. Must be one of {self.options}"
            )
        return value

    def to_dict(self) -> Dict[str, Any]:
        serializable_desc: Dict[str, Any] = {}
        for k, v in self.descriptions.items():
            if isinstance(v, (list, tuple)):
                serializable_desc[k] = list(v)
            else:
                serializable_desc[k] = v
        return {
            "type": self.field_type,
            "name": self.name,
            "description": self.description,
            "required": self.required,
            "options": list(self.options),
            "descriptions": serializable_desc,
            "default": self.default,
            "metadata": self.metadata,
        }


class BooleanField(DecisionField):
    """Binary decision field (e.g., is_safe, allow, requires_escalation)."""

    field_type: ClassVar[str] = "boolean"

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        true_description: str = "True / Positive",
        false_description: str = "False / Negative",
        default: Optional[bool] = None,
        description: str = "",
        required: bool = True,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(description=description, required=required, metadata=metadata)
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"BooleanField threshold must be in [0.0, 1.0], got {threshold}")
        self.threshold: float = float(threshold)
        self.true_description: str = str(true_description).strip()
        self.false_description: str = str(false_description).strip()
        self.default: Optional[bool] = default

    def validate_value(self, value: Any) -> bool:
        if value is None and self.default is not None:
            return self.default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            lower = value.strip().lower()
            if lower in ("true", "1", "yes", "allow", "pass"):
                return True
            if lower in ("false", "0", "no", "deny", "block"):
                return False
        raise TypeError(f"BooleanField {self.name!r} expected bool, got {type(value).__name__} ({value!r})")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.field_type,
            "name": self.name,
            "description": self.description,
            "required": self.required,
            "threshold": self.threshold,
            "true_description": self.true_description,
            "false_description": self.false_description,
            "default": self.default,
            "metadata": self.metadata,
        }


class MultiChoiceField(DecisionField):
    """Multi-label choice field where zero or more options may be selected simultaneously."""

    field_type: ClassVar[str] = "multi_choice"

    def __init__(
        self,
        options: Sequence[str],
        *,
        threshold: float = 0.5,
        descriptions: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
        default: Optional[Sequence[str]] = None,
        description: str = "",
        required: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(description=description, required=required, metadata=metadata)
        if not options:
            raise ValueError("MultiChoiceField requires at least one option")
        cleaned_options = []
        seen = set()
        for opt in options:
            if not isinstance(opt, str) or not opt.strip():
                raise ValueError(f"MultiChoiceField options must be non-empty strings, got {opt!r}")
            cleaned = opt.strip()
            if cleaned in seen:
                raise ValueError(f"Duplicate option in MultiChoiceField: {cleaned!r}")
            seen.add(cleaned)
            cleaned_options.append(cleaned)
        self.options: Tuple[str, ...] = tuple(cleaned_options)
        parsed_descriptions: Dict[str, Union[str, Tuple[str, ...]]] = {}
        for k, v in (descriptions or {}).items():
            if k in seen:
                if isinstance(v, str):
                    parsed_descriptions[k] = v.strip()
                elif isinstance(v, (list, tuple, set)):
                    parsed_descriptions[k] = tuple(str(x).strip() for x in v if str(x).strip())
                else:
                    parsed_descriptions[k] = str(v).strip()
        self.descriptions: Dict[str, Union[str, Tuple[str, ...]]] = parsed_descriptions
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"MultiChoiceField threshold must be in [0.0, 1.0], got {threshold}")
        self.threshold: float = float(threshold)
        if default is not None:
            for item in default:
                if item not in seen:
                    raise ValueError(f"Default item {item!r} not in options {self.options}")
            self.default: Optional[Tuple[str, ...]] = tuple(default)
        else:
            self.default = None

    def validate_value(self, value: Any) -> Tuple[str, ...]:
        if value is None and self.default is not None:
            return self.default
        if not isinstance(value, (list, tuple, set)):
            raise TypeError(
                f"MultiChoiceField {self.name!r} expected sequence of str, got {type(value).__name__}"
            )
        result = []
        for item in value:
            if not isinstance(item, str) or item not in self.options:
                raise ValueError(f"Invalid option {item!r} for MultiChoiceField {self.name!r}")
            result.append(item)
        return tuple(result)

    def to_dict(self) -> Dict[str, Any]:
        serializable_desc: Dict[str, Any] = {}
        for k, v in self.descriptions.items():
            if isinstance(v, (list, tuple)):
                serializable_desc[k] = list(v)
            else:
                serializable_desc[k] = v
        return {
            "type": self.field_type,
            "name": self.name,
            "description": self.description,
            "required": self.required,
            "options": list(self.options),
            "threshold": self.threshold,
            "descriptions": serializable_desc,
            "default": list(self.default) if self.default is not None else None,
            "metadata": self.metadata,
        }


class ScoreField(DecisionField):
    """Continuous bounded numeric rating or risk score in [min_value, max_value]."""

    field_type: ClassVar[str] = "score"

    def __init__(
        self,
        *,
        min_value: float = 0.0,
        max_value: float = 1.0,
        low_description: str = "Minimum boundary",
        high_description: str = "Maximum boundary",
        default: Optional[float] = None,
        description: str = "",
        required: bool = True,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(description=description, required=required, metadata=metadata)
        if min_value >= max_value:
            raise ValueError(
                f"ScoreField min_value ({min_value}) must be strictly less than max_value ({max_value})"
            )
        self.min_value: float = float(min_value)
        self.max_value: float = float(max_value)
        self.low_description: str = str(low_description).strip()
        self.high_description: str = str(high_description).strip()
        if default is not None:
            if not (self.min_value <= default <= self.max_value):
                raise ValueError(
                    f"Default score {default} outside bounds [{self.min_value}, {self.max_value}]"
                )
            self.default: Optional[float] = float(default)
        else:
            self.default = None

    def validate_value(self, value: Any) -> float:
        if value is None and self.default is not None:
            return self.default
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"ScoreField {self.name!r} expected float, got {type(value).__name__}")
        val = float(value)
        if not (self.min_value <= val <= self.max_value):
            raise ValueError(
                f"Score {val} for field {self.name!r} outside bounds [{self.min_value}, {self.max_value}]"
            )
        return val

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.field_type,
            "name": self.name,
            "description": self.description,
            "required": self.required,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "low_description": self.low_description,
            "high_description": self.high_description,
            "default": self.default,
            "metadata": self.metadata,
        }


class SchemaMeta(type):
    """Metaclass that collects DecisionField descriptors and initializes schemas."""

    def __new__(mcs, name: str, bases: tuple[type, ...], attrs: dict[str, Any]) -> SchemaMeta:
        fields: dict[str, DecisionField] = {}
        for base in bases:
            if hasattr(base, "_declared_fields"):
                fields.update(base._declared_fields)

        for key, value in list(attrs.items()):
            if isinstance(value, DecisionField):
                value.bind_name(key)
                fields[key] = value

        attrs["_declared_fields"] = fields
        return super().__new__(mcs, name, bases, attrs)


class DecisionSchema(metaclass=SchemaMeta):
    """Declarative or programmatic container for Reflex decision schemas.

    Subclassing example:
        class AgentTriageSchema(DecisionSchema):
            route = ChoiceField(options=["fs", "db", "net", "human"])
            is_urgent = BooleanField()
            risk_score = ScoreField(min_value=0.0, max_value=1.0)
    """

    _declared_fields: ClassVar[Dict[str, DecisionField]] = {}

    def __init__(
        self,
        fields: Optional[Mapping[str, DecisionField]] = None,
        *,
        schema_name: Optional[str] = None,
        description: str = "",
    ) -> None:
        self.schema_name: str = schema_name or self.__class__.__name__
        self.description: str = description
        self._fields: Dict[str, DecisionField] = dict(self._declared_fields)
        if fields:
            for k, f in fields.items():
                f.bind_name(k)
                self._fields[k] = f
        if not self._fields:
            raise ValueError(f"DecisionSchema {self.schema_name!r} has no fields defined")

    @property
    def fields(self) -> Mapping[str, DecisionField]:
        return dict(self._fields)

    def get_field(self, name: str) -> DecisionField:
        if name not in self._fields:
            raise KeyError(f"Field {name!r} not found in schema {self.schema_name!r}")
        return self._fields[name]

    def validate_decision(self, values: Mapping[str, Any]) -> Dict[str, Any]:
        """Validates a dictionary of evaluated field values against the schema."""
        validated: Dict[str, Any] = {}
        for name, field_def in self._fields.items():
            if name in values and values[name] is not None:
                validated[name] = field_def.validate_value(values[name])
            elif field_def.default is not None:
                validated[name] = field_def.default
            elif field_def.required:
                raise ValueError(
                    f"Required field {name!r} missing in decision values for schema {self.schema_name!r}"
                )
            else:
                validated[name] = None
        return validated

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the complete schema definition into a canonical dictionary."""
        return {
            "schema_name": self.schema_name,
            "description": self.description,
            "fields": {name: f.to_dict() for name, f in sorted(self._fields.items())},
        }

    def canonical_json(self) -> str:
        """Returns deterministic, sorted JSON representation."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def schema_digest(self) -> str:
        """Computes the canonical SHA-256 fingerprint of the schema."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DecisionSchema:
        """Constructs a DecisionSchema instance from serialized dictionary data."""
        schema_name = data.get("schema_name", "DynamicDecisionSchema")
        description = data.get("description", "")
        fields_data = data.get("fields", {})
        fields: Dict[str, DecisionField] = {}

        for name, field_info in fields_data.items():
            ftype = field_info.get("type")
            req = field_info.get("required", True)
            desc = field_info.get("description", "")
            meta = field_info.get("metadata", {})

            if ftype == "choice":
                f = ChoiceField(
                    options=field_info["options"],
                    descriptions=field_info.get("descriptions"),
                    default=field_info.get("default"),
                    description=desc,
                    required=req,
                    metadata=meta,
                )
            elif ftype == "boolean":
                f = BooleanField(
                    threshold=field_info.get("threshold", 0.5),
                    true_description=field_info.get("true_description", "True / Positive"),
                    false_description=field_info.get("false_description", "False / Negative"),
                    default=field_info.get("default"),
                    description=desc,
                    required=req,
                    metadata=meta,
                )
            elif ftype == "multi_choice":
                f = MultiChoiceField(
                    options=field_info["options"],
                    threshold=field_info.get("threshold", 0.5),
                    descriptions=field_info.get("descriptions"),
                    default=field_info.get("default"),
                    description=desc,
                    required=req,
                    metadata=meta,
                )
            elif ftype == "score":
                f = ScoreField(
                    min_value=field_info.get("min_value", 0.0),
                    max_value=field_info.get("max_value", 1.0),
                    low_description=field_info.get("low_description", "Minimum boundary"),
                    high_description=field_info.get("high_description", "Maximum boundary"),
                    default=field_info.get("default"),
                    description=desc,
                    required=req,
                    metadata=meta,
                )
            else:
                raise ValueError(f"Unknown field type {ftype!r} in schema definition")
            fields[name] = f

        return cls(fields=fields, schema_name=schema_name, description=description)

    def to_json_schema(self) -> Dict[str, Any]:
        """Exports the schema into an OpenAPI/JSON Schema compliant structure."""
        properties: Dict[str, Any] = {}
        required: list[str] = []

        for name, f in self._fields.items():
            if f.required:
                required.append(name)
            if isinstance(f, ChoiceField):
                properties[name] = {
                    "type": "string",
                    "enum": list(f.options),
                    "description": f.description,
                }
            elif isinstance(f, BooleanField):
                properties[name] = {
                    "type": "boolean",
                    "description": f.description,
                }
            elif isinstance(f, MultiChoiceField):
                properties[name] = {
                    "type": "array",
                    "items": {"type": "string", "enum": list(f.options)},
                    "description": f.description,
                }
            elif isinstance(f, ScoreField):
                properties[name] = {
                    "type": "number",
                    "minimum": f.min_value,
                    "maximum": f.max_value,
                    "description": f.description,
                }

        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": self.schema_name,
            "description": self.description,
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }


__all__ = [
    "DecisionField",
    "ChoiceField",
    "BooleanField",
    "MultiChoiceField",
    "ScoreField",
    "SchemaMeta",
    "DecisionSchema",
    "_validate_field_name",
]
