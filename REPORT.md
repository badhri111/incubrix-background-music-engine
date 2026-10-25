# Design and results

## Design

- FastAPI serves the HTML/CSS/JavaScript UI and JSON job API from one process.
- Jobs are queued in an in-memory dictionary and executed in daemon threads. This keeps the MVP dependency-free but means jobs are lost on restart and it is not designed for multi-worker production deployment.
- A provider interface separates generation from orchestration. The default procedural provider uses deterministic NumPy/SciPy synthesis and does not call a remote service or use a trained AI model.
- Four independently synthesized stems (drums, bass, harmony, melody) are saved as WAV files. The mix is normalized toward the requested LUFS where the peak limiter permits, and the same master gain curve is applied to each stem to preserve recombination.
- Files are stored under `outputs/{job_id}`. MP3 conversion runs only when requested and FFmpeg exists.

## Verification

The test suite checks API health/readiness, request validation, job/idempotency behavior, generation failures, reproducible procedural stems, exact sample duration, stem downloads, and stem-to-mix recombination. `tests/test_http.py` exercises the HTTP interface with FastAPI's test client. Validation completed with `python -m pytest -q`: 37 passed (with one upstream Starlette `TestClient` deprecation warning).

## Limitations

- The generator is a compact rule-based synthesizer, not AI music generation. Mood mapping is keyword-based and the available profiles are deliberately small.
- Jobs and idempotency keys live only in process memory. Use one application worker; a restart loses job state.
- Loudness is a lightweight K-weighted estimate, not a full production mastering chain. Sample-peak limiting may prevent every result from reaching the selected LUFS target; no oversampled true-peak measurement is claimed.
- Loopable mode applies short edge fades; it does not guarantee a musically seamless loop.
- FFmpeg is optional outside Docker. If unavailable, WAV remains available and the manifest reports MP3 omission.
- Rights and licensing are not independently determined. Users must assess the output and applicable rights themselves.

## Future improvements

Persist jobs and idempotency keys, add cancellation and bounded worker management, improve prompt-to-arrangement mapping, use a full standards-tested loudness meter, and integrate a model provider only after its license and deployment requirements are established.
