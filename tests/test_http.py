"""HTTP integration tests for the public API contract."""

import time

from fastapi.testclient import TestClient

import main


def test_http_health_and_ready():
    with TestClient(main.app) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/ready").json()["status"] == "ready"


def test_http_validation_uses_documented_error_shape():
    with TestClient(main.app) as client:
        response = client.post(
            "/api/v1/music/jobs",
            json={"prompt": "", "duration_s": 30},
        )
    assert response.status_code == 422
    assert set(response.json()) == {"code", "message", "retryable", "details"}


def test_http_idempotency_and_downloads():
    payload = {
        "prompt": "gentle acoustic background music",
        "duration_s": 5,
        "seed": 42,
        "output": {"sample_rate": 48000, "formats": ["wav"]},
    }
    with TestClient(main.app) as client:
        first = client.post(
            "/api/v1/music/jobs",
            json=payload,
            headers={"Idempotency-Key": "http-integration-test"},
        )
        second = client.post(
            "/api/v1/music/jobs",
            json=payload,
            headers={"Idempotency-Key": "http-integration-test"},
        )
        assert first.status_code == 202
        assert first.json()["job_id"] == second.json()["job_id"]

        job_id = first.json()["job_id"]
        for _ in range(200):
            response = client.get(f"/api/v1/jobs/{job_id}")
            job = response.json()
            if job["status"] in ("succeeded", "failed"):
                break
            time.sleep(0.05)
        assert job["status"] == "succeeded"
        assert "mix_mp3" not in job["downloads"]
        assert client.get(job["downloads"]["mix_wav"]).status_code == 200
        for name in ("drums", "bass", "harmony", "melody"):
            assert client.get(job["downloads"][f"stem_{name}"]).status_code == 200
