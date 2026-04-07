"""API request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class SpikeBuffer(BaseModel):
    """Incoming spike buffer for one decode window."""

    spike_times: list[list[float]] = Field(
        ..., description="Per-unit spike timestamps in seconds."
    )
    unit_ids: list[int] = Field(..., description="Per-unit identifiers.")
    t_start: float = Field(default=0.0, ge=0.0)
    t_stop: float = Field(..., gt=0.0)
    n_mc_samples: int = Field(default=50, ge=1)

    @model_validator(mode="after")
    def _validate_shapes(self) -> SpikeBuffer:
        if len(self.spike_times) != len(self.unit_ids):
            msg = "spike_times and unit_ids must have equal length"
            raise ValueError(msg)
        if self.t_stop <= self.t_start:
            msg = "t_stop must be greater than t_start"
            raise ValueError(msg)
        return self


class TrajectoryPoint(BaseModel):
    """One decoded point for a single timestep."""

    t_ms: float
    x: float
    y: float
    x_std: float
    y_std: float
    confidence: float = Field(ge=0.0, le=1.0)


class TrajectoryResponse(BaseModel):
    """Decoded trajectory response payload."""

    timesteps: list[TrajectoryPoint]
