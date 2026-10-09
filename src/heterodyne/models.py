"""Explicit units and component specifications for matched power budgets."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass
class FrequencyTable:
    frequency_hz: object
    values: object
    unit: str
    provenance: str = "user assumption"

    def __post_init__(self):
        self.frequency_hz = np.asarray(self.frequency_hz, dtype=float)
        self.values = np.asarray(self.values, dtype=float)
        if (self.frequency_hz.ndim != 1 or self.values.shape != self.frequency_hz.shape
                or len(self.values) < 2 or not np.all(np.isfinite(self.values))
                or not np.all(np.isfinite(self.frequency_hz))
                or np.any(self.frequency_hz < 0) or np.any(np.diff(self.frequency_hz) <= 0)):
            raise ValueError("Tables require at least two finite, increasing frequencies and matching values")

    def evaluate(self, frequency_hz):
        f = np.asarray(frequency_hz, dtype=float)
        if np.any(f < self.frequency_hz[0]) or np.any(f > self.frequency_hz[-1]):
            raise ValueError("Frequency outside specification table; extrapolation is disabled")
        return np.interp(f, self.frequency_hz, self.values)

    @classmethod
    def from_csv(cls, path, value_column, unit, provenance="user assumption"):
        with Path(path).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        return cls([float(r["frequency_hz"]) for r in rows],
                   [float(r[value_column]) for r in rows], unit, provenance)

    def to_csv(self, path, value_column="value"):
        with Path(path).open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["frequency_hz", value_column])
            writer.writerows(zip(self.frequency_hz, self.values))


Spec = float | FrequencyTable | None


@dataclass
class Component:
    name: str
    gain_db: Spec
    noise_figure_db: Spec = None
    iip3_dbm: Spec = None
    oip3_dbm: Spec = None
    input_p1db_dbm: Spec = None
    output_p1db_dbm: Spec = None
    max_safe_input_dbm: Spec = None
    isolation_db: Spec = None
    temperature_k: float = 290.0
    operating_band_hz: tuple[float, float] | None = None
    kind: Literal["amplifier", "passive", "filter", "switch", "splitter"] = "amplifier"
    provenance: dict[str, str] = field(default_factory=dict)
    state_gain_db: dict[str, Spec] = field(default_factory=dict)

    def __post_init__(self):
        if self.gain_db is None:
            raise ValueError("Gain must be specified")
        if not np.isfinite(self.temperature_k) or self.temperature_k < 0:
            raise ValueError("Temperature must be finite and nonnegative")
        if self.operating_band_hz is not None:
            lo, hi = self.operating_band_hz
            if not 0 <= lo < hi or not np.isfinite(hi):
                raise ValueError("Invalid operating band")
        if self.iip3_dbm is not None and self.oip3_dbm is not None:
            raise ValueError("Specify either input or output IP3")
        if self.input_p1db_dbm is not None and self.output_p1db_dbm is not None:
            raise ValueError("Specify either input or output P1dB")

    def evaluate(self, field_name, frequency_hz, state=None):
        f = np.asarray(frequency_hz, dtype=float)
        if self.operating_band_hz is not None:
            lo, hi = self.operating_band_hz
            if np.any((f < lo) | (f > hi)):
                raise ValueError(f"{self.name}: frequency outside operating band")
        value = getattr(self, field_name)
        if field_name == "gain_db" and state in self.state_gain_db:
            value = self.state_gain_db[state]
        if value is None:
            return np.full(f.shape, np.nan)
        if isinstance(value, FrequencyTable):
            expected = "dBm" if field_name.endswith("dbm") else "dB"
            if value.unit != expected:
                raise ValueError(f"{field_name} requires {expected}, got {value.unit}")
            return value.evaluate(f)
        if not np.isfinite(value):
            raise ValueError(f"{field_name} must be finite")
        return np.full(f.shape, float(value))


@dataclass
class Mixer(Component):
    lo_hz: float = 0.0
    conversion: Literal["down", "up"] = "down"
    sideband: Literal["upper", "lower"] = "upper"
    noise_convention: Literal["SSB", "DSB"] | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.lo_hz <= 0 or not np.isfinite(self.lo_hz):
            raise ValueError("Mixer requires a positive LO frequency")
        if self.conversion not in ("down", "up") or self.sideband not in ("upper", "lower"):
            raise ValueError("Invalid mixer conversion or sideband")
        if self.noise_convention not in ("SSB", "DSB"):
            raise ValueError("Declare SSB or DSB mixer noise figure")

    def output_frequency(self, frequency_hz):
        f = np.asarray(frequency_hz)
        sign = 1 if self.sideband == "upper" else -1
        out = sign * (f - self.lo_hz) if self.conversion == "down" else self.lo_hz + sign * f
        if np.any(out < 0):
            raise ValueError(f"{self.name}: frequencies inconsistent with selected sideband")
        return out


@dataclass
class Chain:
    components: list[Component]
    name: str = "chain"

    def __post_init__(self):
        if not self.components or len({c.name for c in self.components}) != len(self.components):
            raise ValueError("A chain requires components with unique names")


@dataclass
class OperatingState:
    name: str = "receive"
    duty_cycle: float = 1.0
    recovery_time_s: float | None = None

    def __post_init__(self):
        if not np.isfinite(self.duty_cycle) or not 0 <= self.duty_cycle <= 1:
            raise ValueError("Duty cycle must be between zero and one")
        if self.recovery_time_s is not None and (not np.isfinite(self.recovery_time_s) or self.recovery_time_s < 0):
            raise ValueError("Recovery time must be nonnegative")


@dataclass
class IQConversion:
    """Independent branch chains; combined noise requires a supplied correlation."""
    i_chain: Chain
    q_chain: Chain
    combining_convention: Literal["power_sum"] | None = None
    noise_correlation: float | None = None

    def __post_init__(self):
        if self.combining_convention not in (None, "power_sum"):
            raise ValueError("Supported combining convention is power_sum")
        if self.noise_correlation is not None and self.noise_correlation != 0:
            raise ValueError("Only explicitly uncorrelated branch noise is supported")
