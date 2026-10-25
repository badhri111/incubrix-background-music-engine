# IncuBrix Background Music Engine

A small FastAPI application that generates original instrumental audio locally. The current generator is a **rule-based procedural DSP synthesizer**, not a trained or generative AI music model. It produces independent drum, bass, harmony, and melody stems, mixes them, and can encode an MP3 with FFmpeg.

## Requirements

- Python 3.10 or newer (the Docker image uses Python 3.12)
- FFmpeg for MP3 output (WAV generation does not require it)
- Docker and Docker Compose are optional alternatives to a local Python install

## Run locally

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m uvicorn main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) for the UI and [http://localhost:8000/docs](http://localhost:8000/docs) for interactive API documentation. `outputs/` stores generated jobs. No API keys or external services are required. Without FFmpeg, generated jobs still succeed as WAV and include an explanatory warning; `mix_mp3` is omitted.

## Docker Compose

```powershell
Copy-Item .env.example .env
docker compose up --build
```

The service is available at `http://localhost:8000`; outputs persist in the `music_outputs` named volume. Stop it with `Ctrl+C`, or run `docker compose down`. The default image includes FFmpeg.

## Host publicly on Render

Render can build the included Dockerfile from a GitHub repository. Use a persistent disk: generated files and the in-memory job records have different lifetimes, so job IDs/statuses are still lost if the service restarts.

1. Push this project to a **private GitHub repository**. In VS Code, open **Source Control** → **Initialize Repository** → review the files (confirm `.env` and `outputs/` are ignored) → **Publish to GitHub**. Do not publish credentials or generated files.
2. In Render, choose **New** → **Web Service**, connect that repository, and select the branch to deploy.
3. Choose **Docker** as the runtime. Leave the Dockerfile and start command at their defaults. The included container listens on Render's assigned `PORT` (and defaults to port 8000 for local Docker).
4. Set the health check path to `/ready`.
5. Add a persistent disk mounted at `/app/outputs`. Set `OUTPUT_DIR` to `/app/outputs` and `MUSIC_PROVIDER` to `procedural`. Leave `DEBUG` disabled. The other settings have defaults; no API key is required.
6. Choose a service size suitable for CPU audio generation and the persistent-disk option available on your plan, then deploy. Check the Render service logs if the build or health check fails.
7. Open the public `onrender.com` URL. Test `/health`, `/ready`, and then submit a short generation from the UI. Confirm its WAV and stems download before sharing the URL.

Review Render's current pricing and persistent-disk restrictions before choosing a plan. Free/ephemeral storage is not suitable for keeping generated audio across restarts. This MVP has no authentication or rate limiting: anyone who can reach the public site can submit audio jobs and use its CPU and disk. Keep the repository private, monitor usage, and do not expose it to the public until you are comfortable with that risk. For a public production service, add access controls, request limits, and storage cleanup first.

### Free Render demo

For a no-cost demo, follow the steps above but select the **Free** web-service instance, and skip the persistent disk. Build from the Dockerfile as described. The app can still generate and serve audio while running; however, Render spins down an idle free service and its local files are ephemeral. Generated mixes/stems and the in-memory job list will be lost when the service spins down, restarts, or redeploys. The first visit after idle may take about a minute to wake up. Free services are intended for testing and hobby projects, not production; check Render's current [free-instance limits](https://render.com/docs/free) before deploying.

Use short demo tracks and download the mix, stems, and manifest as soon as generation completes. The app has no authentication or rate limiting, so anyone with the public URL can submit jobs. Do not use this free demo for private audio, guaranteed availability, or durable storage.

## API

Submit a job (returns HTTP 202):

```powershell
curl.exe -X POST http://localhost:8000/api/v1/music/jobs `
  -H "Content-Type: application/json" `
  -H "Idempotency-Key: demo-001" `
  -d '{\"prompt\":\"calm acoustic music building into an uplifting finish\",\"duration_s\":30,\"sections\":[{\"start_s\":0,\"mood\":\"calm\"},{\"start_s\":15,\"mood\":\"uplifting\"}],\"tempo_bpm\":100,\"seed\":42,\"output\":{\"sample_rate\":48000,\"target_lufs\":-16,\"formats\":[\"wav\",\"mp3\"]}}'
```

Poll the returned ID at `GET /api/v1/jobs/{job_id}`. A successful job includes download links for the mix, the separate stems, and the JSON manifest. Duration must be from 5 to 300 seconds; tempo, when supplied, must be from 40 to 220 BPM. Errors use `code`, `message`, `retryable`, and `details` fields. Reusing an `Idempotency-Key` returns the original job.

The optional `output.formats` accepts `wav` and/or `mp3`; `wav` is required and MP3 is created only when requested and FFmpeg is available. WAV mix and stems are fixed at 48 kHz.

## Batch requests

Pass a JSON array of request objects (or one request object) to the batch helper. It submits, polls, and downloads each successful result:

```powershell
python scripts\batch_test.py .\requests.json --base-url http://localhost:8000
```

Per-job output and a combined `batch_summary.json` are written to `batch_results/`.

## Tests

```powershell
python -m pytest -q
```

The tests use the local deterministic mock provider for API jobs and exercise the procedural synthesizer directly. No network service or paid credential is used.

## Generation and rights limitations

Mood words select from a small set of hand-authored musical profiles. Sections affect energy and rhythmic intensity. The engine synthesizes separate drum, bass, chord/harmony, and melody tracks using NumPy/SciPy; it does not understand arbitrary natural-language instructions like a general music model. The manifest records this method, seed, requested details, timings, and any FFmpeg warning.

The application does not verify copyright ownership, licensing, or commercial clearance. It makes no promise that generated output is royalty-free or cleared for commercial use. WAV files are floating-point audio; the stored stems share the mix master curve and their manifest gains reproduce the mix. See [SOURCES.md](./SOURCES.md) and [REPORT.md](./REPORT.md).
