from dataclasses import dataclass
from pathlib import Path
import os

@dataclass
class StudioConfig:
    data_dir: Path
    port: int = 8765
    max_bytes: int = 200 * 1024 * 1024
    max_seconds: float = 3600

    def __post_init__(self):
        self.data_dir = Path(self.data_dir).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def default(cls):
        base = Path(os.environ.get('LOCALAPPDATA', Path.home() / '.local/share'))
        return cls(Path(os.environ.get('EIGHTD_STUDIO_DATA_DIR', base / '8DEngineStudio')))

    def resolve(self, relative: str) -> Path:
        path = (self.data_dir / relative).resolve()
        if not path.is_relative_to(self.data_dir):
            raise ValueError('Invalid asset path')
        return path
