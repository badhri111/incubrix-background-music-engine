import numpy as np
from models import MusicJobRequest
from providers.base import BaseMusicProvider, GenerationResult
from providers.procedural import ProceduralSynthesizerProvider


class AIModelProvider(BaseMusicProvider):
    """
    Provider seam for a future external model. No external AI service is implemented.
    """
    def __init__(self):
        self.fallback_engine = ProceduralSynthesizerProvider()

    def generate(self, request: MusicJobRequest, sample_rate: int = 48000) -> GenerationResult:
        result = self.fallback_engine.generate(request, sample_rate)
        result.warnings.append(
            "External AI music generation is not implemented; local procedural DSP was used."
        )
        result.generation_method_info["provider"] = "AIModelProvider (procedural fallback)"
        return result

    def regenerate_single_stem(
        self,
        request: MusicJobRequest,
        stem_name: str,
        new_seed: int,
        tempo_bpm: int,
        key: str,
        sample_rate: int = 48000
    ) -> np.ndarray:
        return self.fallback_engine.regenerate_single_stem(
            request, stem_name, new_seed, tempo_bpm, key, sample_rate
        )
