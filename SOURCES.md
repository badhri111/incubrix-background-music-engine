# Sources, dependencies, and rights

## Runtime dependencies

| Dependency | Purpose | License |
|---|---|---|
| FastAPI | HTTP API and OpenAPI serving | MIT |
| Uvicorn | ASGI server | BSD-3-Clause |
| Pydantic | Request/response validation | MIT |
| NumPy | Audio arrays and synthesis | BSD-3-Clause |
| SciPy | DSP filters, WAV I/O, and signal processing | BSD-3-Clause |
| Requests | Optional local/URL voiceover WAV loading in the engine | Apache-2.0 |
| python-dotenv | Load local settings from `.env` | BSD-3-Clause |
| FFmpeg | Optional MP3 encoding; included by the Docker image | LGPL-2.1-or-later or GPL-2-or-later depending on build options |

`pytest` (MIT) and `httpx` (BSD-3-Clause) are used for tests. The project does not bundle recordings, loops, external media, model weights, or call an external music-generation service.

## Generated audio rights

The output is produced by procedural synthesis. This project does not make a legal determination, provide a rights warranty, or claim verified commercial clearance for generated files. Check dependency and system-package licenses for the exact versions and builds you distribute.

## References

- FastAPI: https://github.com/fastapi/fastapi
- Uvicorn: https://github.com/encode/uvicorn
- Pydantic: https://github.com/pydantic/pydantic
- NumPy: https://github.com/numpy/numpy
- SciPy: https://github.com/scipy/scipy
- Requests: https://github.com/psf/requests
- FFmpeg licensing: https://ffmpeg.org/legal.html
