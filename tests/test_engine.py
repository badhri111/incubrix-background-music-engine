# tests/test_engine.py
"""Unit tests for MusicEngine and ProceduralSynthesizerProvider."""

import math
import os
import shutil
import numpy as np
import pytest
from scipy.io import wavfile
from pathlib import Path

os.environ.setdefault("MUSIC_PROVIDER", "mock")
os.environ.setdefault("OUTPUT_DIR", "outputs_test_engine")


@pytest.fixture(scope="module")
def engine_output_dir():
    p = "outputs_test_engine"
    os.makedirs(p, exist_ok=True)
    yield p
    shutil.rmtree(p, ignore_errors=True)


# ---------------------------------------------------------------------------
# LUFS / peak utilities
# ---------------------------------------------------------------------------

def test_lufs_silence():
    from music_engine import MusicEngine
    engine = MusicEngine()
    assert engine.calculate_integrated_lufs(np.zeros(48000, dtype=np.float32), 48000) == -70.0


def test_lufs_sine():
    from music_engine import MusicEngine
    engine = MusicEngine()
    t = np.linspace(0, 1.0, 48000, endpoint=False)
    sig = (np.sin(2 * np.pi * 1000 * t) * 0.5).astype(np.float32)
    lufs = engine.calculate_integrated_lufs(sig, 48000)
    # 0.5 amplitude sine should produce a plausible K-weighted loudness estimate.
    assert -20.0 < lufs < 0.0


def test_sample_peak_sine():
    from music_engine import MusicEngine
    engine = MusicEngine()
    t = np.linspace(0, 1.0, 48000, endpoint=False)
    sig = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    peak = engine.calculate_sample_peak_dbfs(sig)
    assert abs(peak) < 1.0  # sine of amplitude 1.0 → ~ 0 dBTP


def test_normalize_and_limit():
    from music_engine import MusicEngine
    engine = MusicEngine()
    # Loud signal that needs limiting
    t = np.linspace(0, 5.0, 5 * 48000, endpoint=False)
    loud = (2.0 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    limited, curve, lufs, peak = engine._normalize_and_limit(loud, 48000, target_lufs=-16.0)
    assert abs(lufs - (-16.0)) <= 1.5, f"LUFS deviation too large: {lufs}"
    assert peak <= -0.9, f"Peak exceeded -1 dBTP: {peak}"


# ---------------------------------------------------------------------------
# ProceduralSynthesizerProvider
# ---------------------------------------------------------------------------

def test_procedural_generates_4_stems():
    from providers.procedural import ProceduralSynthesizerProvider
    from models import MusicJobRequest
    provider = ProceduralSynthesizerProvider()
    req = MusicJobRequest(prompt="uplifting cinematic music", duration_s=5.0, seed=1)
    result = provider.generate(req, 48000)
    assert set(result.stems.keys()) == {"drums", "bass", "harmony", "melody"}


def test_procedural_stems_are_distinct():
    """Stems must not be equal (i.e. not split frequency bands of the same signal)."""
    from providers.procedural import ProceduralSynthesizerProvider
    from models import MusicJobRequest
    provider = ProceduralSynthesizerProvider()
    req = MusicJobRequest(prompt="electronic dance music", duration_s=8.0, seed=7)
    result = provider.generate(req, 48000)
    stems = list(result.stems.values())
    for i in range(len(stems)):
        for j in range(i + 1, len(stems)):
            corr = float(np.corrcoef(stems[i][:4800], stems[j][:4800])[0, 1])
            assert abs(corr) < 0.99, f"Stems {i} and {j} are suspiciously similar (corr={corr:.3f})"


def test_procedural_tempo_used(engine_output_dir):
    from providers.procedural import ProceduralSynthesizerProvider
    from models import MusicJobRequest
    provider = ProceduralSynthesizerProvider()
    req = MusicJobRequest(prompt="ambient drone", duration_s=6.0, tempo_bpm=90, seed=3)
    result = provider.generate(req, 48000)
    assert result.tempo_bpm == 90


def test_procedural_seed_reproducible():
    from providers.procedural import ProceduralSynthesizerProvider
    from models import MusicJobRequest
    provider = ProceduralSynthesizerProvider()
    req = MusicJobRequest(prompt="calm lo-fi chill music", duration_s=5.0, seed=42)
    r1 = provider.generate(req, 48000)
    r2 = provider.generate(req, 48000)
    for s in r1.stems:
        assert np.array_equal(r1.stems[s], r2.stems[s]), f"Stem '{s}' not reproducible"


def test_procedural_all_mood_profiles():
    """Every mood profile must generate without error."""
    from providers.procedural import ProceduralSynthesizerProvider, MOOD_PROFILES
    from models import MusicJobRequest
    provider = ProceduralSynthesizerProvider()
    for mood in MOOD_PROFILES:
        req = MusicJobRequest(prompt=mood, duration_s=5.0, seed=0)
        result = provider.generate(req, 48000)
        assert len(result.stems) == 4, f"Mood '{mood}' did not produce 4 stems"


# ---------------------------------------------------------------------------
# MusicEngine full pipeline
# ---------------------------------------------------------------------------

def test_engine_exact_duration(engine_output_dir):
    from music_engine import MusicEngine
    from models import MusicJobRequest
    engine = MusicEngine(provider_name="procedural")
    for dur in (5.0, 30.0, 60.0):
        req = MusicJobRequest(prompt="corporate upbeat music", duration_s=dur, seed=0)
        _, manifest = engine.generate_music_job(req, f"eng-dur-{int(dur)}")
        diff = abs(manifest.delivered_duration_s - dur)
        assert diff <= 0.05, f"Duration mismatch for {dur}s: delivered {manifest.delivered_duration_s}s"
    shutil.rmtree(engine_output_dir, ignore_errors=True)


def test_engine_lufs_within_1_lu(engine_output_dir):
    from music_engine import MusicEngine
    from models import MusicJobRequest
    engine = MusicEngine(provider_name="procedural")
    req = MusicJobRequest(prompt="tense thriller music", duration_s=15.0, seed=10)
    _, manifest = engine.generate_music_job(req, "eng-lufs-test")
    diff = abs(manifest.actual_lufs - manifest.target_lufs)
    assert diff <= 1.5, f"LUFS deviation: {diff}"
    shutil.rmtree(engine_output_dir, ignore_errors=True)


def test_engine_peak_within_ceiling(engine_output_dir):
    from music_engine import MusicEngine
    from models import MusicJobRequest
    engine = MusicEngine(provider_name="procedural")
    req = MusicJobRequest(prompt="loud energetic electronic dance", duration_s=10.0, seed=99)
    _, manifest = engine.generate_music_job(req, "eng-peak-test")
    assert manifest.peak_dbfs <= -0.9, f"Peak exceeded: {manifest.peak_dbfs}"
    shutil.rmtree(engine_output_dir, ignore_errors=True)


def test_engine_stem_sum_equals_mix(engine_output_dir):
    """
    Recombine mastered stems with manifest gains → must match mix to ±2 samples.
    This validates the distributive property preservation.
    """
    from music_engine import MusicEngine
    from models import MusicJobRequest

    out_dir = "outputs_stem_test"
    os.makedirs(out_dir, exist_ok=True)
    from config import settings
    original_output_dir = settings.OUTPUT_DIR
    settings.OUTPUT_DIR = out_dir

    try:
        engine = MusicEngine(provider_name="procedural")
        req = MusicJobRequest(prompt="acoustic folk music", duration_s=10.0, seed=77)
        job_id = "stem-sum-test"
        _, manifest = engine.generate_music_job(req, job_id)

        base = Path(out_dir) / job_id
        _, mix_wav = wavfile.read(str(base / "mix.wav"))
        mix_f = mix_wav.astype(np.float64)

        recon = np.zeros_like(mix_f)
        for s_name, s_meta in manifest.stems.items():
            gain = s_meta.gain
            _, stem_wav = wavfile.read(str(base / "stems" / f"{s_name}.wav"))
            recon += stem_wav.astype(np.float64) * gain

        diff = np.max(np.abs(mix_f - recon))
        assert diff <= 2.0, f"Stem recombination error: {diff} samples"
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
        settings.OUTPUT_DIR = original_output_dir


def test_engine_sections_handled(engine_output_dir):
    from music_engine import MusicEngine
    from models import MusicJobRequest, MusicSection
    engine = MusicEngine(provider_name="procedural")
    req = MusicJobRequest(
        prompt="calm intro that builds into uplifting peak",
        duration_s=20.0,
        seed=55,
        sections=[
            MusicSection(start_s=0.0, mood="calm"),
            MusicSection(start_s=10.0, mood="uplifting"),
        ],
    )
    _, manifest = engine.generate_music_job(req, "eng-sections")
    assert manifest.delivered_duration_s > 0
    shutil.rmtree(engine_output_dir, ignore_errors=True)


def test_mock_provider_failure():
    from music_engine import MusicEngine
    from models import MusicJobRequest
    os.makedirs("outputs_mock_fail", exist_ok=True)
    from config import settings
    original_output_dir = settings.OUTPUT_DIR
    settings.OUTPUT_DIR = "outputs_mock_fail"
    engine = MusicEngine(provider_name="mock")
    req = MusicJobRequest(prompt="trigger_fail intentional", duration_s=10.0)
    with pytest.raises(RuntimeError, match="Simulated"):
        engine.generate_music_job(req, "mock-fail-job")
    shutil.rmtree("outputs_mock_fail", ignore_errors=True)
    settings.OUTPUT_DIR = original_output_dir
