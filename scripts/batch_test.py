"""Submit JSON music requests, wait for completion, and save their artifacts."""

import argparse
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


def read_requests(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError("Input must be a JSON object, an array of objects, or JSON Lines.")
    return data


def request_json(url: str, method: str = "GET", payload: dict | None = None, key: str | None = None):
    headers = {"Accept": "application/json"}
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode("utf-8")
    if key:
        headers["Idempotency-Key"] = key
    req = Request(url, data=body, headers=headers, method=method)
    with urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def download(url: str, destination: Path) -> None:
    with urlopen(url, timeout=60) as response:
        destination.write_bytes(response.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requests", type=Path, help="JSON array/object or JSON Lines input file")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--results-dir", type=Path, default=Path("batch_results"))
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/") + "/"
    requests_data = read_requests(args.requests)
    results = []
    for index, payload in enumerate(requests_data, start=1):
        job = request_json(
            urljoin(base_url, "api/v1/music/jobs"),
            method="POST",
            payload=payload,
            key=f"batch-{args.requests.resolve()}-{index}",
        )
        job_id = job["job_id"]
        deadline = time.monotonic() + args.timeout
        while job["status"] not in ("succeeded", "failed"):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Job {job_id} did not finish before the timeout.")
            time.sleep(args.poll_interval)
            job = request_json(urljoin(base_url, f"api/v1/jobs/{job_id}"))

        job_dir = args.results_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        downloaded = []
        if job["status"] == "succeeded":
            for name, relative_url in job["downloads"].items():
                target = job_dir / Path(relative_url).name
                download(urljoin(base_url, relative_url.lstrip("/")), target)
                downloaded.append(str(target))
        summary = {
            "request": payload,
            "job": job,
            "downloaded_files": downloaded,
        }
        (job_dir / "result.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        results.append(summary)
        print(f"{index}/{len(requests_data)}: {job_id} — {job['status']}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "batch_summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return 0 if all(result["job"]["status"] == "succeeded" for result in results) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        raise SystemExit(f"Batch run failed: {exc}") from exc
