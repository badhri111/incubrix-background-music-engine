from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any
import numpy as np
from models import MusicJobRequest


@dataclass
class GenerationResult:
    stems: Dict[str, np.ndarray]
    stem_gains: Dict[str, float]
    tempo_bpm: int
    key: str
    time_signature: str = "4/4"
    beat_times: List[float] = field(default_factory=list)
    bar_times: List[float] = field(default_factory=list)
    generation_method_info: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class BaseMusicProvider(ABC):
    @abstractmethod
    def generate(self, request: MusicJobRequest, sample_rate: int) -> GenerationResult:
        """Generate all stems and musical grid according to the request."""
        pass

    @abstractmethod
    def regenerate_single_stem(
        self,
        request: MusicJobRequest,
        stem_name: str,
        new_seed: int,
        tempo_bpm: int,
        key: str,
        sample_rate: int,
    ) -> np.ndarray:
        """Regenerate a single stem with a new seed while preserving the musical piece's grid."""
        pass
