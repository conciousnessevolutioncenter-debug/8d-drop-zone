from pydantic import BaseModel, Field, ConfigDict, field_validator
from eightd_engine.dsp import panning_preset_names

class MixSettings(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    rotation_cpm: float = Field(default=5.78, ge=0.1, le=30)
    room_size: float = Field(default=0.18, ge=0, le=0.5)
    crossover_hz: float = Field(default=150, ge=40, le=300)
    motion_depth: float = Field(default=1, ge=0, le=1)
    panning_preset: str = 'binaural_8d'

    @field_validator('panning_preset')
    @classmethod
    def valid_preset(cls, value):
        if value not in panning_preset_names():
            raise ValueError('Unknown spatial preset')
        return value
