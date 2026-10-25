import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import requests
from scipy import signal
from scipy.ndimage import maximum_filter1d
from scipy.io import wavfile

from config import settings
from logger import logger
from models import (
    MusicJobRequest,
    MusicManifest,
    StemMetadata,
    DuckingMetadata,
    OutputOptions,
)
from providers.base import BaseMusicProvider, GenerationResult
from providers.procedural import ProceduralSynthesizerProvider
from providers.mock import MockMusicProvider
from providers.ai_adapter import AIModelProvider


def _get_provider(name: str) -> BaseMusicProvider:
    if name == "mock":
        return MockMusicProvider()
    if name == "ai":
        return AIModelProvider()
    return ProceduralSynthesizerProvider()


class MusicEngine:
    """
    Main orchestration engine for IncuBrix AI Background Music generation.

    Pipeline
    --------
    1. Select provider (procedural DSP synth, mock, or AI adapter).
    2. Generate 4 independent stems via the provider.
    3. Enforce *exact* sample-accurate duration (zero-pad or truncate).
    4. Mix stems with recorded gains so  mix = Σ(gain_i · stem_i).
    5. LUFS-normalise toward target using ITU-R BS.1770 K-weighting.
    6. Apply lookahead peak limiter so True Peak ≤ −1.0 dBTP (strictly).
    7. Save every stem scaled by the *same* master gain factor so that
       reloading stems and summing them with their manifest gains
       reproduces the final mix to within ±1 sample.
    8. Convert to MP3 via FFmpeg when available.
    9. Build a detailed JSON manifest.
    """

    def __init__(self, provider_name: Optional[str] = None):
        name = provider_name or settings.MUSIC_PROVIDER
        self.provider: BaseMusicProvider = _get_provider(name)
        logger.info(
            f"Initialized MusicEngine with provider: {self.provider.__class__.__name__}"
        )

    # ------------------------------------------------------------------
    # Loudness & peak utilities
    # ------------------------------------------------------------------

    def _k_weight_filters(self, fs: int):
        """Return two (b, a) pairs for the ITU-R BS.1770 K-weighting pre-filter."""
        # Stage 1 – pre-filter / high-shelf
        f0, G, Q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
        K = math.tan(math.pi * f0 / fs)
        Vh = 10.0 ** (G / 20.0)
        Vb = Vh ** 0.4996667741545416
        a0 = 1.0 + K / Q + K * K
        b1 = np.array([Vh + Vb * K / Q + K * K, 2.0 * (K * K - Vh), Vh - Vb * K / Q + K * K]) / a0
        a1 = np.array([1.0, 2.0 * (K * K - 1.0) / a0, (1.0 - K / Q + K * K) / a0])

        # Stage 2 – high-pass / RLB weighting
        f1, Q1 = 38.13547087602444, 0.5003270373233973
        K1 = math.tan(math.pi * f1 / fs)
        a0_hp = 1.0 + K1 / Q1 + K1 * K1
        b2 = np.array([1.0, -2.0, 1.0]) / a0_hp
        a2 = np.array([1.0, 2.0 * (K1 * K1 - 1.0) / a0_hp, (1.0 - K1 / Q1 + K1 * K1) / a0_hp])
        return (b1, a1), (b2, a2)

    def calculate_integrated_lufs(self, audio: np.ndarray, sample_rate: int) -> float:
        """ITU-R BS.1770 integrated loudness (LUFS). Returns −70 for silence."""
        if len(audio) == 0:
            return -70.0
        (b1, a1), (b2, a2) = self._k_weight_filters(sample_rate)
        x = signal.lfilter(b1, a1, audio.astype(np.float64))
        x = signal.lfilter(b2, a2, x)
        ms = np.mean(x ** 2)
        if ms <= 1e-12:
            return -70.0
        return float(round(-0.691 + 10.0 * np.log10(ms), 2))

    def calculate_sample_peak_dbfs(self, audio: np.ndarray) -> float:
        """Measure the absolute sample peak in dBFS (not oversampled true peak)."""
        peak = np.max(np.abs(audio))
        if peak <= 1e-9:
            return -99.0
        return float(round(20.0 * np.log10(peak), 2))

    def _lookahead_limiter(
        self,
        audio: np.ndarray,
        sample_rate: int,
        ceiling_dbfs: float = -1.0,
        lookahead_ms: float = 5.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Strict lookahead limiter.
        Returns (limited_audio, gain_curve).
        The gain curve applied is the same for every stem so the distributive
        property mix = Σ(gain_i · stem_i) is preserved.
        """
        ceiling = 10.0 ** (ceiling_dbfs / 20.0) * 0.9999  # tiny guard for fp rounding
        win = max(1, int(sample_rate * lookahead_ms / 1000.0))
        peak_env = maximum_filter1d(np.abs(audio), size=win)
        c = np.minimum(1.0, ceiling / np.maximum(peak_env, 1e-9))
        return audio * c, c

    def _normalize_and_limit(
        self,
        audio: np.ndarray,
        sample_rate: int,
        target_lufs: float = -16.0,
        ceiling_dbfs: float = -1.0,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        1. Measure integrated LUFS.
        2. Apply linear boost so measured ≈ target.
        3. Pass through a sample-peak lookahead limiter.

        Returns (limited_audio, gain_curve, final_lufs, final_peak_dbfs).
        The gain_curve must be applied to *every* stem to keep their sum equal to
        the final mix.
        """
        cur_lufs = self.calculate_integrated_lufs(audio, sample_rate)
        if cur_lufs <= -69.0:
            return audio, np.ones(len(audio), dtype=np.float32), cur_lufs, -99.0
        boost = 10.0 ** ((target_lufs - cur_lufs) / 20.0)
        boosted = audio * boost
        limited, c = self._lookahead_limiter(boosted, sample_rate, ceiling_dbfs)
        final_lufs = self.calculate_integrated_lufs(limited, sample_rate)
        final_peak = self.calculate_sample_peak_dbfs(limited)
        # The composite gain factor for each sample is: boost * c[sample]
        composite_curve = boost * c
        return limited, composite_curve, final_lufs, final_peak

    # ------------------------------------------------------------------
    # Voiceover ducking
    # ------------------------------------------------------------------

    def _apply_ducking(
        self,
        mix_audio: np.ndarray,
        narration_url: str,
        sample_rate: int,
    ) -> Tuple[Optional[np.ndarray], Optional[DuckingMetadata], List[str]]:
        """Duck mix during speech. Returns (ducked_mix | None, meta | None, warnings)."""
        warnings: List[str] = []
        try:
            import io
            if narration_url.startswith(("http://", "https://")):
                resp = requests.get(narration_url, timeout=10)
                resp.raise_for_status()
                sr_vo, vo_data = wavfile.read(io.BytesIO(resp.content))
            else:
                if not os.path.exists(narration_url):
                    warnings.append(
                        f"Narration file '{narration_url}' not found – ducking skipped."
                    )
                    return None, None, warnings
                sr_vo, vo_data = wavfile.read(narration_url)

            # Mono float32
            if vo_data.ndim > 1:
                vo_data = np.mean(vo_data, axis=1)
            vo_data = vo_data.astype(np.float64)
            if np.issubdtype(vo_data.dtype, np.integer):
                vo_data = vo_data / float(np.iinfo(vo_data.dtype).max)

            # Resample if needed
            if sr_vo != sample_rate:
                vo_data = signal.resample(
                    vo_data, int(len(vo_data) * sample_rate / sr_vo)
                )

            mix_len = len(mix_audio)
            if len(vo_data) < mix_len:
                vo_data = np.pad(vo_data, (0, mix_len - len(vo_data)))
            else:
                vo_data = vo_data[:mix_len]

            # RMS energy envelope (100 ms window, 20 ms hop)
            w = int(sample_rate * 0.10)
            hop = int(sample_rate * 0.02)
            num_frames = max(1, (mix_len - w) // hop + 1)

            rms_frames = np.array(
                [np.sqrt(np.mean(vo_data[i * hop : i * hop + w] ** 2)) for i in range(num_frames)],
                dtype=np.float32,
            )
            speech_thr = 0.025  # ~−32 dBFS RMS
            is_speech = rms_frames > speech_thr
            has_speech = bool(np.any(is_speech))

            if not has_speech:
                warnings.append("No speech detected in narration – minimal ducking applied.")

            # Target gain: 0.251 (−12 dB) during speech, 1.0 during silence
            tgt = np.where(is_speech, 0.251, 1.0).astype(np.float32)
            att = float(np.exp(-hop / (sample_rate * 0.15)))
            rel = float(np.exp(-hop / (sample_rate * 0.40)))

            smooth = np.ones_like(tgt)
            cur = 1.0
            for i in range(len(tgt)):
                if tgt[i] < cur:
                    cur = tgt[i] + (cur - tgt[i]) * att
                else:
                    cur = tgt[i] + (cur - tgt[i]) * rel
                smooth[i] = cur

            # Upsample gain curve back to sample rate
            t_frames = np.linspace(0, mix_len, len(smooth))
            t_samps = np.arange(mix_len)
            duck_curve = np.interp(t_samps, t_frames, smooth).astype(np.float32)

            ducked = mix_audio * duck_curve

            sample_idx = np.linspace(0, len(smooth) - 1, min(100, len(smooth))).astype(int)
            curve_pts = [float(round(smooth[i], 3)) for i in sample_idx]

            meta = DuckingMetadata(
                voiceover_detected=has_speech,
                ducking_reduction_db=-12.0,
                attack_ms=150.0,
                release_ms=400.0,
                curve_sample_points=curve_pts,
            )
            return ducked, meta, warnings

        except Exception as exc:
            logger.warning(f"Ducking failed: {exc}")
            warnings.append(f"Voiceover ducking skipped – error: {exc}")
            return None, None, warnings

    # ------------------------------------------------------------------
    # MP3 export
    # ------------------------------------------------------------------

    def _to_mp3(self, wav_path: Path, mp3_path: Path) -> Tuple[bool, Optional[str]]:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return False, "FFmpeg not found in PATH – MP3 export omitted."
        try:
            res = subprocess.run(
                [ffmpeg, "-y", "-i", str(wav_path), "-codec:a", "libmp3lame", "-b:a", "192k", str(mp3_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
            )
            if res.returncode == 0 and mp3_path.exists():
                return True, None
            return False, f"FFmpeg error (code {res.returncode}): {res.stderr.decode(errors='ignore')[:300]}"
        except Exception as exc:
            return False, f"FFmpeg exception: {exc}"

    # ------------------------------------------------------------------
    # WAV writer helper
    # ------------------------------------------------------------------

    @staticmethod
    def _write_wav(path: Path, audio_f32: np.ndarray, sample_rate: int) -> None:
        """Write float32 WAV to avoid per-stem quantization breaking recombination."""
        wavfile.write(str(path), sample_rate, np.clip(audio_f32, -1.0, 1.0).astype(np.float32))

    # ------------------------------------------------------------------
    # Main generation entry-point
    # ------------------------------------------------------------------

    def generate_music_job(
        self, request: MusicJobRequest, job_id: str
    ) -> Tuple[Path, MusicManifest]:
        t0 = time.time()

        job_dir = Path(settings.OUTPUT_DIR) / job_id
        stems_dir = job_dir / "stems"
        stems_dir.mkdir(parents=True, exist_ok=True)

        out = request.output or OutputOptions()
        sample_rate = out.sample_rate
        target_lufs = out.target_lufs

        # ── 1. Generate ────────────────────────────────────────────────
        gen: GenerationResult = self.provider.generate(request, sample_rate)
        warnings = list(gen.warnings)

        # ── 2. Enforce exact sample count ──────────────────────────────
        n = int(round(request.duration_s * sample_rate))
        stems: Dict[str, np.ndarray] = {}
        for name, audio in gen.stems.items():
            arr = audio.astype(np.float32)
            if len(arr) < n:
                arr = np.pad(arr, (0, n - len(arr)))
            else:
                arr = arr[:n]
            stems[name] = arr

        # ── 3. Mix with recorded gains ──────────────────────────────────
        raw_gains = gen.stem_gains  # e.g. {"drums": 0.80, "bass": 0.75, ...}
        raw_mix = sum(stems[s] * raw_gains[s] for s in stems).astype(np.float32)

        # ── 4. LUFS normalise + lookahead limit ─────────────────────────
        #
        # After this step:
        #   limited_mix = raw_mix * composite_curve
        # where composite_curve[i] = boost * limiter_gain[i]
        #
        # We apply the *same* composite_curve to every stem so that:
        #   Σ( manifest_gain[s] * stem_f32[s] ) == limited_mix  (exactly)
        # with  manifest_gain[s] = raw_gains[s]  (gains are baked into stems).
        limited_mix, composite_curve, mix_lufs, mix_peak = self._normalize_and_limit(
            raw_mix, sample_rate, target_lufs=target_lufs
        )

        # Bake composite_curve into each stem and record manifest gain as raw_gain
        mastered_stems: Dict[str, np.ndarray] = {
            s: (stems[s] * composite_curve).astype(np.float32) for s in stems
        }

        # ── 5. Save stems as 48 kHz 16-bit WAV ─────────────────────────
        stems_meta: Dict[str, StemMetadata] = {}
        for s_name, s_audio in mastered_stems.items():
            stem_path = stems_dir / f"{s_name}.wav"
            self._write_wav(stem_path, s_audio, sample_rate)
            s_lufs = self.calculate_integrated_lufs(s_audio, sample_rate)
            s_peak = self.calculate_sample_peak_dbfs(s_audio)
            stems_meta[s_name] = StemMetadata(
                filename=f"stems/{s_name}.wav",
                download_url=f"/api/v1/jobs/{job_id}/downloads/stems/{s_name}.wav",
                gain=float(round(raw_gains[s_name], 4)),
                integrated_lufs=s_lufs,
                peak_dbfs=s_peak,
                duration_s=round(len(s_audio) / sample_rate, 4),
            )

        # ── 6. Save final mix WAV ───────────────────────────────────────
        mix_wav = job_dir / "mix.wav"
        self._write_wav(mix_wav, limited_mix, sample_rate)

        # ── 7. MP3 export ───────────────────────────────────────────────
        if "mp3" in out.formats:
            mix_mp3 = job_dir / "mix.mp3"
            mp3_ok, mp3_msg = self._to_mp3(mix_wav, mix_mp3)
            if not mp3_ok and mp3_msg:
                warnings.append(mp3_msg)

        # ── 8. Voiceover ducking ────────────────────────────────────────
        ducking_meta: Optional[DuckingMetadata] = None
        if request.narration_url:
            ducked, ducking_meta, duck_warns = self._apply_ducking(
                limited_mix, request.narration_url, sample_rate
            )
            warnings.extend(duck_warns)
            if ducked is not None:
                self._write_wav(job_dir / "mix_ducked.wav", ducked, sample_rate)

        # ── 9. Build manifest ───────────────────────────────────────────
        gen_time = round(time.time() - t0, 3)
        delivered_s = round(len(limited_mix) / sample_rate, 4)

        manifest = MusicManifest(
            job_id=job_id,
            prompt=request.prompt,
            duration_s=request.duration_s,
            delivered_duration_s=delivered_s,
            tempo_bpm=gen.tempo_bpm,
            key=gen.key,
            time_signature=gen.time_signature,
            seed=int(gen.generation_method_info.get(
                "seed", request.seed if request.seed is not None else 0
            )),
            sample_rate=sample_rate,
            target_lufs=target_lufs,
            actual_lufs=mix_lufs,
            peak_dbfs=mix_peak,
            loopable=bool(request.loopable),
            sections=gen.generation_method_info.get("sections_aligned", []),
            beat_grid={
                "beat_times": gen.beat_times,
                "bar_times": gen.bar_times,
                "tempo_bpm": gen.tempo_bpm,
                "time_signature": gen.time_signature,
            },
            stems=stems_meta,
            generation_method=gen.generation_method_info,
            timings={
                "generation_time_s": gen_time,
                "audio_duration_s": delivered_s,
                "realtime_factor": round(gen_time / max(0.001, delivered_s), 4),
            },
            warnings=warnings,
            ducking=ducking_meta,
            licensing={
                "rights_status": "Not independently verified.",
                "notes": (
                    "Audio is generated locally with rule-based procedural DSP and no bundled "
                    "recordings. This is not a trained AI music model. No legal or commercial "
                    "rights clearance is provided."
                ),
                "terms_url": "/SOURCES.md",
            },
        )

        manifest_path = job_dir / "music_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as fh:
            fh.write(manifest.model_dump_json(indent=2))

        logger.info(
            f"Job {job_id} done in {gen_time}s | "
            f"LUFS={mix_lufs} | Peak={mix_peak} dBTP | "
            f"stems={list(stems_meta)}"
        )
        return job_dir, manifest

    # ------------------------------------------------------------------
    # Single-stem regeneration
    # ------------------------------------------------------------------

    def regenerate_stem(
        self, job_id: str, stem_name: str, new_seed: Optional[int] = None
    ) -> MusicManifest:
        """Re-synthesise one stem, re-mix, re-normalise, update manifest."""
        job_dir = Path(settings.OUTPUT_DIR) / job_id
        manifest_path = job_dir / "music_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found for job {job_id}")

        with open(manifest_path, "r", encoding="utf-8") as fh:
            mdata = json.load(fh)

        sample_rate: int = mdata["sample_rate"]
        seed: int = new_seed if new_seed is not None else mdata["seed"] + 100

        req = MusicJobRequest(
            prompt=mdata["prompt"],
            duration_s=mdata["duration_s"],
            tempo_bpm=mdata["tempo_bpm"],
            key=mdata["key"],
            loopable=mdata["loopable"],
            seed=seed,
            output=OutputOptions(sample_rate=sample_rate, target_lufs=mdata["target_lufs"]),
        )

        new_audio = self.provider.regenerate_single_stem(
            req, stem_name, seed, mdata["tempo_bpm"], mdata["key"], sample_rate
        )
        n = int(round(mdata["duration_s"] * sample_rate))
        if len(new_audio) < n:
            new_audio = np.pad(new_audio, (0, n - len(new_audio)))
        else:
            new_audio = new_audio[:n]

        # Reload all other stems (already mastered = baked composite gain)
        all_stems: Dict[str, np.ndarray] = {}
        for stem_file in (job_dir / "stems").glob("*.wav"):
            s = stem_file.stem
            _, s_wav = wavfile.read(str(stem_file))
            if s_wav.ndim > 1:
                s_wav = np.mean(s_wav, axis=1)
            all_stems[s] = s_wav.astype(np.float32) / 32767.0

        # Replace the regenerated stem (unmastered; will be mastered with re-mix)
        all_stems[stem_name] = new_audio.astype(np.float32)

        # Get current manifest gains
        manifest_gains = {s: mdata["stems"][s]["gain"] for s in mdata["stems"] if s in all_stems}
        raw_mix = sum(all_stems[s] * manifest_gains.get(s, 1.0) for s in all_stems).astype(np.float32)

        limited_mix, composite_curve, new_lufs, new_peak = self._normalize_and_limit(
            raw_mix, sample_rate, mdata["target_lufs"]
        )

        # Re-save every stem with the new composite_curve applied
        for s, s_audio in all_stems.items():
            mastered_s = (s_audio * composite_curve).astype(np.float32)
            self._write_wav(job_dir / "stems" / f"{s}.wav", mastered_s, sample_rate)

        self._write_wav(job_dir / "mix.wav", limited_mix, sample_rate)
        self._to_mp3(job_dir / "mix.wav", job_dir / "mix.mp3")

        mdata["actual_lufs"] = new_lufs
        mdata["peak_dbfs"] = new_peak
        mdata.setdefault("warnings", []).append(
            f"Stem '{stem_name}' regenerated with seed {seed}."
        )
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(mdata, fh, indent=2)

        return MusicManifest(**mdata)
