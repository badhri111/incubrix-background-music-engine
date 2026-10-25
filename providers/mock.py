import numpy as np
from models import MusicJobRequest
from providers.base import BaseMusicProvider, GenerationResult


class MockMusicProvider(BaseMusicProvider):
    """
    Mock provider for fast deterministic tests, failure injection, and simulating timeouts.
    """
    def generate(self, request: MusicJobRequest, sample_rate: int = 48000) -> GenerationResult:
        if "trigger_fail" in request.prompt.lower():
            raise RuntimeError("Simulated upstream model generation failure.")

        num_samples = int(round(request.duration_s * sample_rate))
        t = np.linspace(0, request.duration_s, num_samples, False)

        # Generate simple distinct pure tones for stems
        drums = (0.2 * np.sin(2.0 * np.pi * 60.0 * t)).astype(np.float32)
        bass = (0.3 * np.sin(2.0 * np.pi * 110.0 * t)).astype(np.float32)
        harmony = (0.2 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
        melody = (0.2 * np.sin(2.0 * np.pi * 880.0 * t)).astype(np.float32)

        stems = {
            "drums": drums,
            "bass": bass,
            "harmony": harmony,
            "melody": melody
        }
        stem_gains = {
            "drums": 0.8,
            "bass": 0.75,
            "harmony": 0.65,
            "melody": 0.70
        }
        tempo_bpm = request.tempo_bpm or 120
        beat_s = 60.0 / tempo_bpm
        beat_times = [round(i * beat_s, 4) for i in range(int(request.duration_s / beat_s))]
        bar_times = [round(i * 4 * beat_s, 4) for i in range(int(len(beat_times) / 4))]

        return GenerationResult(
            stems=stems,
            stem_gains=stem_gains,
            tempo_bpm=tempo_bpm,
            key=request.key or "C Major",
            time_signature="4/4",
            beat_times=beat_times,
            bar_times=bar_times,
            generation_method_info={
                "provider": "MockMusicProvider",
                "method": "Synthetic Test Fixture",
                "version": "1.0.0"
            },
            warnings=["Generated via MockMusicProvider test fixture."]
        )

    def regenerate_single_stem(
        self,
        request: MusicJobRequest,
        stem_name: str,
        new_seed: int,
        tempo_bpm: int,
        key: str,
        sample_rate: int = 48000
    ) -> np.ndarray:
        num_samples = int(round(request.duration_s * sample_rate))
        t = np.linspace(0, request.duration_s, num_samples, False)
        return (0.25 * np.sin(2.0 * np.pi * 500.0 * t)).astype(np.float32)
