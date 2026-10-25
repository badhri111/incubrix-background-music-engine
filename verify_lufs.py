import numpy as np
from models import MusicJobRequest
from music_engine import MusicEngine

engine = MusicEngine()
prompts = [
    'cinematic epic movie score',
    'lo-fi chill study beats',
    'corporate modern presentation music',
    'electronic dance club synthwave',
    'acoustic warm organic guitar folk',
    'tense thriller suspense heartbeat',
    'playful cheerful happy kids game',
    'ambient space meditative drone',
    'chill relaxing peaceful evening',
    'uplifting optimistic commercial pop'
]
all_ok = True
for i, p in enumerate(prompts):
    req = MusicJobRequest(prompt=p, duration_s=10.0, seed=42)
    _, m = engine.generate_music_job(req, f'verify-{i:02d}')
    lufs_diff = abs(m.actual_lufs - m.target_lufs)
    ok = lufs_diff <= 1.5 and m.peak_dbfs <= -0.9
    status = "OK" if ok else "FAIL"
    print(f'{p[:22]:22s} | LUFS {m.actual_lufs:6.2f} | Peak {m.peak_dbfs:5.2f} dBFS | diff {lufs_diff:.2f} | {status}')
    if not ok:
        all_ok = False

print('All OK!' if all_ok else 'Some FAILED')
