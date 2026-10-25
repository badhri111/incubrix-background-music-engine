# tests/test_api.py
"""
API-level integration tests using direct function calls (no HTTP transport).
All tests run against the mock provider — no paid APIs, no network required.
"""

import os
import time
from pathlib import Path

import pytest

# Patch env before importing main
os.environ["MUSIC_PROVIDER"] = "mock"
os.environ["OUTPUT_DIR"] = "outputs_test"

import main  # noqa: E402  – must come after env setup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wait_job(job_id: str, timeout: float = 30.0) -> dict:
    """Poll the in-memory job store until status is terminal."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with main._lock:
            job = main._jobs.get(job_id)
        if job and job["status"] in ("succeeded", "failed"):
            return job
        time.sleep(0.15)
    raise TimeoutError(f"Job {job_id} did not finish in {timeout}s")


def _submit(prompt: str, duration_s: float, **extra) -> dict:
    """Submit a job and return the initial response dict."""
    from models import MusicJobRequest
    req = MusicJobRequest(prompt=prompt, duration_s=duration_s, **extra)
    response = main.create_job(req, idempotency_key=None)
    return response.model_dump()


# ---------------------------------------------------------------------------
# 1. Health & readiness endpoints
# ---------------------------------------------------------------------------

def test_health():
    result = main.health()
    assert result["status"] == "ok"


def test_ready(tmp_path):
    # Temporarily point to a writable dir
    orig = main.settings.OUTPUT_DIR
    main.settings.OUTPUT_DIR = str(tmp_path / "outputs_ready")
    try:
        result = main.ready()
        assert result["status"] == "ready"
    finally:
        main.settings.OUTPUT_DIR = orig


# ---------------------------------------------------------------------------
# 2. Valid & invalid generation requests
# ---------------------------------------------------------------------------

VALID_REQ = dict(prompt="calm acoustic guitar", duration_s=10, seed=42)


def test_create_valid_job():
    from models import MusicJobRequest
    req = MusicJobRequest(**VALID_REQ)
    resp = main.create_job(req, idempotency_key=None)
    assert resp.job_id
    assert resp.status in ("queued", "running")


def test_invalid_duration_too_short():
    from fastapi import HTTPException
    from models import MusicJobRequest
    req = MusicJobRequest(prompt="music", duration_s=2)
    with pytest.raises(HTTPException) as exc_info:
        main.create_job(req, idempotency_key=None)
    assert exc_info.value.status_code == 422


def test_invalid_duration_too_long():
    from fastapi import HTTPException
    from models import MusicJobRequest
    req = MusicJobRequest(prompt="music", duration_s=400)
    with pytest.raises(HTTPException) as exc_info:
        main.create_job(req, idempotency_key=None)
    assert exc_info.value.status_code == 422


def test_invalid_tempo():
    from fastapi import HTTPException
    from models import MusicJobRequest
    req = MusicJobRequest(prompt="music", duration_s=10, tempo_bpm=10)
    with pytest.raises(HTTPException) as exc_info:
        main.create_job(req, idempotency_key=None)
    assert exc_info.value.status_code == 422


def test_invalid_section_beyond_duration():
    from fastapi import HTTPException
    from models import MusicJobRequest, MusicSection
    req = MusicJobRequest(
        prompt="music", duration_s=10,
        sections=[MusicSection(start_s=999, mood="calm")]
    )
    with pytest.raises(HTTPException) as exc_info:
        main.create_job(req, idempotency_key=None)
    assert exc_info.value.status_code == 422


def test_unknown_job():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        main.get_job("does-not-exist-xyz")
    assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# 3. Correct output duration
# ---------------------------------------------------------------------------

def test_output_duration():
    target_dur = 15.0
    resp = _submit("calm acoustic", target_dur, seed=1)
    job = _wait_job(resp["job_id"])
    assert job["status"] == "succeeded"
    delivered = job["manifest"]["delivered_duration_s"]
    assert abs(delivered - target_dur) <= 0.05, f"Duration mismatch: {delivered} vs {target_dur}"


# ---------------------------------------------------------------------------
# 4. Existence and duration of all four stems
# ---------------------------------------------------------------------------

def test_all_stems_exist():
    from scipy.io import wavfile

    target_dur = 8.0
    resp = _submit("uplifting pop music", target_dur, seed=2)
    job = _wait_job(resp["job_id"])
    assert job["status"] == "succeeded"

    job_id = job["job_id"]
    manifest = job["manifest"]
    stem_names = list(manifest["stems"].keys())
    assert len(stem_names) >= 3, "Expected at least 3 stems"

    for s in stem_names:
        stem_dur = manifest["stems"][s]["duration_s"]
        assert abs(stem_dur - target_dur) <= 0.05, f"Stem '{s}' duration off: {stem_dur}"

        stem_path = Path("outputs_test") / job_id / "stems" / f"{s}.wav"
        assert stem_path.exists(), f"Stem file missing: {stem_path}"
        sr, data = wavfile.read(str(stem_path))
        assert sr == 48000
        actual_dur = len(data) / sr
        assert abs(actual_dur - target_dur) <= 0.05


# ---------------------------------------------------------------------------
# 5. Stem recombination reproduces the mix (± 2 samples)
# ---------------------------------------------------------------------------

def test_stem_recombination():
    import numpy as np
    from scipy.io import wavfile

    resp = _submit("lo-fi chill beats", 12.0, seed=77)
    job = _wait_job(resp["job_id"])
    assert job["status"] == "succeeded"

    job_id = job["job_id"]
    manifest = job["manifest"]
    stems_info = manifest["stems"]

    base = Path("outputs_test") / job_id
    _, mix_wav = wavfile.read(str(base / "mix.wav"))

    recombined = np.zeros(len(mix_wav), dtype=np.float64)
    for s_name, s_meta in stems_info.items():
        gain = s_meta["gain"]
        _, stem_wav = wavfile.read(str(base / "stems" / f"{s_name}.wav"))
        recombined += stem_wav.astype(np.float64) * gain

    diff = np.max(np.abs(mix_wav.astype(np.float64) - recombined))
    assert diff <= 2.0, (
        f"Stem recombination error {diff} is > 2 samples. "
        f"stems={list(stems_info)}"
    )


# ---------------------------------------------------------------------------
# 6. Idempotency – same key returns the same job
# ---------------------------------------------------------------------------

def test_idempotency_key():
    from models import MusicJobRequest
    req = MusicJobRequest(**VALID_REQ)
    idem_key = "test-idem-key-abc999"
    resp1 = main.create_job(req, idempotency_key=idem_key)
    resp2 = main.create_job(req, idempotency_key=idem_key)
    assert resp1.job_id == resp2.job_id, "Same idempotency key must return the same job"


def test_different_idempotency_keys():
    from models import MusicJobRequest
    req = MusicJobRequest(**VALID_REQ)
    r1 = main.create_job(req, idempotency_key="uniq-111")
    r2 = main.create_job(req, idempotency_key="uniq-222")
    assert r1.job_id != r2.job_id


# ---------------------------------------------------------------------------
# 7. Job status transitions & failure handling
# ---------------------------------------------------------------------------

def test_job_status_transitions():
    resp = _submit("corporate background music", 8.0, seed=5)
    job = _wait_job(resp["job_id"])
    assert job["status"] == "succeeded"
    assert job["progress"] == 100
    assert job["downloads"] is not None


def test_generation_failure_handling():
    """Mock provider raises RuntimeError on 'trigger_fail' prompt."""
    resp = _submit("trigger_fail intentional test", 10.0)
    job = _wait_job(resp["job_id"])
    assert job["status"] == "failed"
    assert job["error"] is not None
    err = job["error"]
    assert err["code"] == "GENERATION_FAILED"
    assert err["retryable"] is True


def test_failed_job_does_not_crash_server():
    """Server stays healthy after a job failure."""
    _submit("trigger_fail crash test", 5.0)
    result = main.health()
    assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# 8. No external API keys required
# ---------------------------------------------------------------------------

def test_no_api_keys_required():
    """Full generation must work with no paid API keys configured."""
    assert not os.environ.get("AI_MUSIC_API_KEY"), "AI_MUSIC_API_KEY must not be set in test mode"
    resp = _submit("calm ambient music", 6.0, seed=99)
    job = _wait_job(resp["job_id"])
    assert job["status"] == "succeeded"
    assert "mix_wav" in job["downloads"]


# ---------------------------------------------------------------------------
# 9. Download links are present for all expected files
# ---------------------------------------------------------------------------

def test_download_links_present():
    resp = _submit("tense thriller score", 7.0, seed=10)
    job = _wait_job(resp["job_id"])
    assert job["status"] == "succeeded"
    links = job["downloads"]
    assert "mix_wav" in links
    # All stem links present
    for stem in ["drums", "bass", "harmony", "melody"]:
        assert f"stem_{stem}" in links, f"Missing stem link for '{stem}'"


def test_manifest_job_endpoint():
    from fastapi import HTTPException
    resp = _submit("playful happy music", 5.0, seed=11)
    job = _wait_job(resp["job_id"])
    assert job["status"] == "succeeded"
    job_id = job["job_id"]
    # Manifest file must exist
    manifest_path = Path("outputs_test") / job_id / "music_manifest.json"
    assert manifest_path.exists()
    # manifest endpoint must return FileResponse (no error)
    response = main.get_manifest(job_id)
    assert response.path  # FileResponse has a .path attribute
    # Also verify get_job returns a manifest dict
    fetched = main.get_job(job_id)
    assert fetched.manifest is not None
    assert "tempo_bpm" in fetched.manifest
    assert "licensing" in fetched.manifest
