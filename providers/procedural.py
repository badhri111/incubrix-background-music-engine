import math
import re
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from scipy import signal
from models import MusicJobRequest
from providers.base import BaseMusicProvider, GenerationResult

# MIDI note to frequency helper
def midi_to_freq(midi_note: float) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))

# Standard Musical Scales and Chords
SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
}

ROOT_NOTES = {
    "C": 60, "C#": 61, "Db": 61, "D": 62, "D#": 63, "Eb": 63,
    "E": 64, "F": 65, "F#": 66, "Gb": 66, "G": 67, "G#": 68,
    "Ab": 68, "A": 69, "A#": 70, "Bb": 70, "B": 71
}

# Known Artists / Songs for safety check
KNOWN_ARTISTS = [
    "ed sheeran", "taylor swift", "beatles", "queen", "michael jackson",
    "drake", "eminem", "billie eilish", "the weeknd", "coldplay",
    "adele", "beyonce", "rihanna", "bruno mars", "led zeppelin",
    "pink floyd", "nirvana", "metallica", "ac/dc", "daft punk"
]

VOCAL_KEYWORDS = ["sing", "singing", "lyrics", "vocals", "singer", "vocal", "rap", "words"]

# Mood Profiles mapping prompts to musical parameters
MOOD_PROFILES = {
    "cinematic": {
        "bpm": 84, "key": "D Minor", "root": 62, "scale": "minor",
        "progressions": [[0, 8, 5, 7], [0, 3, 7, 8]], # Dm - Bb - Gm - Am
        "drum_density": 0.5, "bass_type": "deep_sub", "harmony_type": "orchestral_strings",
        "melody_style": "heroic", "energy": 0.8
    },
    "lo-fi": {
        "bpm": 78, "key": "F Major", "root": 65, "scale": "major",
        "progressions": [[2, 7, 0, 9]], # Gm7 - C7 - Fmaj7 - Dm7
        "drum_density": 0.6, "bass_type": "warm_sine", "harmony_type": "mellow_keys",
        "melody_style": "relaxed", "energy": 0.5
    },
    "chill": {
        "bpm": 80, "key": "C Major", "root": 60, "scale": "major",
        "progressions": [[0, 9, 5, 7]], # C - Am - F - G
        "drum_density": 0.5, "bass_type": "warm_sine", "harmony_type": "mellow_keys",
        "melody_style": "relaxed", "energy": 0.5
    },
    "corporate": {
        "bpm": 120, "key": "C Major", "root": 60, "scale": "major",
        "progressions": [[0, 5, 9, 7], [0, 7, 9, 5]], # C - F - Am - G
        "drum_density": 0.75, "bass_type": "plucked_clean", "harmony_type": "acoustic_pluck",
        "melody_style": "bright", "energy": 0.7
    },
    "uplifting": {
        "bpm": 118, "key": "G Major", "root": 67, "scale": "major",
        "progressions": [[0, 7, 9, 5]], # G - D - Em - C
        "drum_density": 0.8, "bass_type": "driving", "harmony_type": "acoustic_pluck",
        "melody_style": "bright", "energy": 0.85
    },
    "electronic": {
        "bpm": 126, "key": "A Minor", "root": 69, "scale": "minor",
        "progressions": [[0, 8, 5, 7], [0, 5, 8, 7]], # Am - F - Dm - Em
        "drum_density": 0.9, "bass_type": "synth_saw", "harmony_type": "supersaw_pad",
        "melody_style": "arpeggiated", "energy": 0.9
    },
    "acoustic": {
        "bpm": 98, "key": "E Major", "root": 64, "scale": "major",
        "progressions": [[0, 9, 5, 7], [0, 7, 9, 5]], # E - C#m - A - B
        "drum_density": 0.5, "bass_type": "warm_sine", "harmony_type": "acoustic_strum",
        "melody_style": "folk_flute", "energy": 0.6
    },
    "tense": {
        "bpm": 105, "key": "C Minor", "root": 60, "scale": "minor",
        "progressions": [[0, 1, 0, 6], [0, 8, 6, 7]], # Cm - C#dim - Cm - F#dim
        "drum_density": 0.7, "bass_type": "distorted_sub", "harmony_type": "dark_drone",
        "melody_style": "suspense", "energy": 0.75
    },
    "playful": {
        "bpm": 132, "key": "F Major", "root": 65, "scale": "major",
        "progressions": [[0, 9, 5, 7]], # F - Dm - Bb - C
        "drum_density": 0.7, "bass_type": "bounce", "harmony_type": "marimba_pluck",
        "melody_style": "bouncy", "energy": 0.8
    },
    "ambient": {
        "bpm": 68, "key": "A Minor", "root": 69, "scale": "minor",
        "progressions": [[0, 5, 8, 7]], # Am - Dm - F - Em
        "drum_density": 0.2, "bass_type": "deep_sub", "harmony_type": "evolving_pad",
        "melody_style": "floating", "energy": 0.35
    }
}


class ProceduralSynthesizerProvider(BaseMusicProvider):
    """
    Algorithmic multi-track procedural synthesizer generating 4 distinct instrument stems:
    - drums (kick, snare, hi-hats, percs)
    - bass (sub/synth/plucked bassline)
    - harmony (pads/arpeggios/chords)
    - melody (lead themes and motifs)
    
    Local math-based DSP synthesis using no bundled audio recordings.
    """

    def _analyze_prompt(self, prompt: str) -> Tuple[Dict[str, Any], List[str]]:
        warnings = []
        p_lower = prompt.lower()

        # Check for artist or song mentions
        for artist in KNOWN_ARTISTS:
            if re.search(r"\b" + re.escape(artist) + r"\b", p_lower):
                warnings.append(
                    f"Prompt referenced artist or song '{artist}'. "
                    "The engine uses procedural synthesis and does not intentionally reproduce "
                    "a named artist or song."
                )

        # Check for vocal requests
        for vocal_kw in VOCAL_KEYWORDS:
            if re.search(r"\b" + re.escape(vocal_kw) + r"\b", p_lower):
                warnings.append(
                    f"Vocal request keyword '{vocal_kw}' detected. Engine strictly generates 100% instrumental music."
                )
                break

        # Check for multilingual or mood keywords
        matched_mood = None
        # Multilingual mapping
        lang_maps = {
            "tranquila": "chill", "calme": "chill", "ruhige": "chill", "calmo": "chill",
            "alegre": "playful", "joyeux": "playful", "fröhlich": "playful",
            "sombre": "tense", "oscuro": "tense", "düster": "tense",
            "acustica": "acoustic", "acoustique": "acoustic", "akustisch": "acoustic",
            "emouvant": "cinematic", "cinematografico": "cinematic"
        }
        for foreign_word, mapped in lang_maps.items():
            if foreign_word in p_lower:
                matched_mood = mapped
                warnings.append(f"Detected multilingual keyword '{foreign_word}' mapped to mood '{mapped}'.")
                break

        if not matched_mood:
            # Check mood profiles in prompt
            for mood_name in MOOD_PROFILES.keys():
                if mood_name in p_lower:
                    matched_mood = mood_name
                    break

        if not matched_mood:
            # Secondary keywords
            if any(w in p_lower for w in ["calm", "relax", "peace", "soft", "gentle"]):
                matched_mood = "chill"
            elif any(w in p_lower for w in ["dark", "creepy", "fear", "suspense", "danger"]):
                matched_mood = "tense"
            elif any(w in p_lower for w in ["happy", "fun", "game", "kids"]):
                matched_mood = "playful"
            elif any(w in p_lower for w in ["energy", "dance", "club", "beat"]):
                matched_mood = "electronic"
            elif any(w in p_lower for w in ["guitar", "folk", "organic"]):
                matched_mood = "acoustic"
            elif any(w in p_lower for w in ["movie", "epic", "dramatic"]):
                matched_mood = "cinematic"
            elif any(w in p_lower for w in ["tech", "business", "presentation", "startup"]):
                matched_mood = "corporate"
            else:
                matched_mood = "uplifting"
                warnings.append("No specific mood category detected; generated versatile uplifting instrumental.")

        # Check contradictory moods
        contradictions = [("calm", "aggressive"), ("slow", "fast"), ("sad", "happy"), ("tense", "relax")]
        for w1, w2 in contradictions:
            if w1 in p_lower and w2 in p_lower:
                warnings.append(f"Contradictory mood keywords ('{w1}' vs '{w2}') detected; dynamically blended.")

        profile = dict(MOOD_PROFILES[matched_mood])
        profile["detected_mood"] = matched_mood
        return profile, warnings

    def _generate_drum_stem(
        self,
        num_samples: int,
        sample_rate: int,
        tempo_bpm: int,
        beat_times: List[float],
        drum_density: float,
        sections_info: List[Dict[str, Any]],
        rng: np.random.RandomState,
        loopable: bool,
    ) -> np.ndarray:
        drum_audio = np.zeros(num_samples, dtype=np.float32)
        beat_duration_s = 60.0 / tempo_bpm
        
        # Pre-synthesize drum one-shots
        # 1. Kick drum
        kick_len = int(sample_rate * 0.35)
        t_k = np.linspace(0, 0.35, kick_len, False)
        # Exponential pitch sweep 140Hz -> 48Hz
        freq_k = 48.0 + (140.0 - 48.0) * np.exp(-t_k * 30.0)
        phase_k = 2.0 * np.pi * np.cumsum(freq_k) / sample_rate
        kick = np.sin(phase_k) * np.exp(-t_k * 10.0)
        kick += 0.3 * np.sin(2.0 * phase_k) * np.exp(-t_k * 18.0)
        # Click transient
        kick[: int(sample_rate * 0.005)] += rng.uniform(-0.4, 0.4, int(sample_rate * 0.005))
        kick = np.clip(kick, -1.0, 1.0).astype(np.float32)

        # 2. Snare drum
        snare_len = int(sample_rate * 0.25)
        t_s = np.linspace(0, 0.25, snare_len, False)
        snare_tone = np.sin(2.0 * np.pi * 185.0 * t_s) * np.exp(-t_s * 18.0)
        # Filtered noise
        noise = rng.uniform(-1.0, 1.0, snare_len)
        # BPF around 2000-5000Hz
        sos_snare = signal.butter(4, [1500, 6000], btype="bandpass", fs=sample_rate, output="sos")
        snare_noise = signal.sosfilt(sos_snare, noise) * np.exp(-t_s * 15.0)
        snare = (0.4 * snare_tone + 0.6 * snare_noise).astype(np.float32)

        # 3. Closed Hi-hat
        hh_len = int(sample_rate * 0.05)
        t_hh = np.linspace(0, 0.05, hh_len, False)
        hh_noise = rng.uniform(-1.0, 1.0, hh_len)
        sos_hh = signal.butter(4, 7500, btype="highpass", fs=sample_rate, output="sos")
        hihat = (signal.sosfilt(sos_hh, hh_noise) * np.exp(-t_hh * 60.0) * 0.6).astype(np.float32)

        # 4. Open Hi-hat
        ohh_len = int(sample_rate * 0.2)
        t_ohh = np.linspace(0, 0.2, ohh_len, False)
        ohh_noise = rng.uniform(-1.0, 1.0, ohh_len)
        open_hihat = (signal.sosfilt(sos_hh, ohh_noise) * np.exp(-t_ohh * 18.0) * 0.5).astype(np.float32)

        # 5. Cymbal crash (for section transitions and musical ending)
        crash_len = min(int(sample_rate * 2.5), num_samples)
        t_cr = np.linspace(0, 2.5, crash_len, False)
        cr_noise = rng.uniform(-1.0, 1.0, crash_len)
        crash = (signal.sosfilt(sos_hh, cr_noise) * np.exp(-t_cr * 2.2) * 0.5).astype(np.float32)

        def add_sample(target: np.ndarray, sample: np.ndarray, start_idx: int, gain: float):
            if start_idx >= len(target) or gain <= 0.001:
                return
            end_idx = min(start_idx + len(sample), len(target))
            target[start_idx:end_idx] += sample[: end_idx - start_idx] * gain

        # Trigger section crash cymbals on section changes
        for sec in sections_info:
            s_idx = int(sec["effective_start_s"] * sample_rate)
            if s_idx < num_samples - sample_rate:
                add_sample(drum_audio, crash, s_idx, 0.5 * sec["energy"])

        # Populate drum beats along the beat grid
        for beat_num, b_time in enumerate(beat_times):
            b_idx = int(b_time * sample_rate)
            if b_idx >= num_samples:
                break

            # Find active section energy
            sec_energy = 0.7
            for sec in sections_info:
                if b_time >= sec["effective_start_s"]:
                    sec_energy = sec["energy"]

            bar_beat = beat_num % 4 # 0, 1, 2, 3 in 4/4 meter
            is_intro = b_time < beat_duration_s * 4 and len(sections_info) > 1

            # Kick pattern (Beats 0 and 2, or syncopated)
            kick_gain = 0.0
            if bar_beat == 0:
                kick_gain = 0.9 * sec_energy
            elif bar_beat == 2 and drum_density > 0.3:
                kick_gain = 0.8 * sec_energy
            elif bar_beat == 1 and drum_density > 0.7 and rng.random() > 0.6:
                kick_gain = 0.7 * sec_energy

            if kick_gain > 0:
                add_sample(drum_audio, kick, b_idx, kick_gain)

            # Snare pattern (Beats 1 and 3)
            snare_gain = 0.0
            if not is_intro:
                if bar_beat in (1, 3):
                    snare_gain = 0.85 * sec_energy
                elif bar_beat == 3 and drum_density > 0.8 and rng.random() > 0.5:
                    # Snare ghost note
                    ghost_idx = b_idx + int(sample_rate * (beat_duration_s * 0.5))
                    add_sample(drum_audio, snare, ghost_idx, 0.4 * sec_energy)

            if snare_gain > 0:
                add_sample(drum_audio, snare, b_idx, snare_gain)

            # Hi-hat pattern (8th notes)
            hh_gain = 0.7 * sec_energy
            add_sample(drum_audio, hihat, b_idx, hh_gain)
            # Offbeat 8th note
            offbeat_idx = b_idx + int(sample_rate * (beat_duration_s * 0.5))
            if offbeat_idx < num_samples:
                if bar_beat in (1, 3) and rng.random() > 0.65 and drum_density > 0.5:
                    add_sample(drum_audio, open_hihat, offbeat_idx, 0.6 * sec_energy)
                else:
                    add_sample(drum_audio, hihat, offbeat_idx, 0.55 * sec_energy)

        # Musical resolution at end
        if not loopable and num_samples > int(sample_rate * 2.0):
            # Final crash and resolving kick on last bar line
            last_bar_time = beat_times[max(0, len(beat_times) - 4)]
            last_bar_idx = int(last_bar_time * sample_rate)
            add_sample(drum_audio, kick, last_bar_idx, 0.9)
            add_sample(drum_audio, crash, last_bar_idx, 0.7)

        return np.clip(drum_audio, -1.0, 1.0)

    def _generate_bass_stem(
        self,
        num_samples: int,
        sample_rate: int,
        tempo_bpm: int,
        beat_times: List[float],
        root_midi: int,
        scale: str,
        progression: List[int],
        bass_type: str,
        sections_info: List[Dict[str, Any]],
        rng: np.random.RandomState,
    ) -> np.ndarray:
        bass_audio = np.zeros(num_samples, dtype=np.float32)
        beat_duration_s = 60.0 / tempo_bpm
        scale_intervals = SCALES.get(scale, SCALES["major"])
        
        # Bass root octave (MIDI 36 = C2)
        bass_root_offset = (root_midi % 12) + 36

        for beat_num, b_time in enumerate(beat_times):
            b_idx = int(b_time * sample_rate)
            if b_idx >= num_samples:
                break

            bar_idx = beat_num // 4
            prog_step = progression[bar_idx % len(progression)]
            scale_step = scale_intervals[prog_step % len(scale_intervals)]
            note_midi = bass_root_offset + scale_step

            # Rhythm groove: trigger on beat 0 and beat 2 or 8th notes
            bar_beat = beat_num % 4
            if bar_beat in (0, 2):
                freq = midi_to_freq(note_midi)
                note_dur = beat_duration_s * 1.6
                note_len = min(int(sample_rate * note_dur), num_samples - b_idx)
                if note_len <= 0:
                    continue

                t = np.linspace(0, note_dur, note_len, False)
                # Dual oscillator: sub sine + harmonic saw/triangle
                sub = np.sin(2.0 * np.pi * freq * t)
                if bass_type in ("synth_saw", "distorted_sub"):
                    saw = signal.sawtooth(2.0 * np.pi * freq * t)
                    wave = 0.5 * sub + 0.5 * saw
                else:
                    triangle = signal.sawtooth(2.0 * np.pi * freq * t, width=0.5)
                    wave = 0.7 * sub + 0.3 * triangle

                # Filter envelope
                sos = signal.butter(2, min(freq * 4.0, sample_rate * 0.45), btype="lowpass", fs=sample_rate, output="sos")
                wave_filtered = signal.sosfilt(sos, wave)
                # Amplitude ADSR envelope
                env = np.exp(-t * (2.5 if bass_type != "deep_sub" else 1.2))
                # Fade in 10ms to prevent click
                fade_in_len = min(int(sample_rate * 0.01), note_len)
                env[:fade_in_len] *= np.linspace(0.0, 1.0, fade_in_len)

                bass_audio[b_idx : b_idx + note_len] += (wave_filtered * env * 0.75).astype(np.float32)

        return np.clip(bass_audio, -1.0, 1.0)

    def _generate_harmony_stem(
        self,
        num_samples: int,
        sample_rate: int,
        tempo_bpm: int,
        beat_times: List[float],
        root_midi: int,
        scale: str,
        progression: List[int],
        harmony_type: str,
        sections_info: List[Dict[str, Any]],
        rng: np.random.RandomState,
    ) -> np.ndarray:
        harmony_audio = np.zeros(num_samples, dtype=np.float32)
        beat_duration_s = 60.0 / tempo_bpm
        scale_intervals = SCALES.get(scale, SCALES["major"])
        bar_duration_s = beat_duration_s * 4.0
        
        # Chord voicings around octave 4 (MIDI 60)
        chord_root_base = (root_midi % 12) + 60

        total_bars = int(math.ceil(len(beat_times) / 4.0))
        for bar in range(total_bars):
            start_beat = bar * 4
            if start_beat >= len(beat_times):
                break
            b_time = beat_times[start_beat]
            b_idx = int(b_time * sample_rate)
            if b_idx >= num_samples:
                break

            prog_idx = progression[bar % len(progression)]
            # Form triad or 7th chord from scale
            note1 = chord_root_base + scale_intervals[prog_idx % len(scale_intervals)]
            note2 = chord_root_base + scale_intervals[(prog_idx + 2) % len(scale_intervals)]
            note3 = chord_root_base + scale_intervals[(prog_idx + 4) % len(scale_intervals)]
            chord_notes = [note1, note2, note3]
            if harmony_type in ("mellow_keys", "evolving_pad"):
                # Add 7th
                note4 = chord_root_base + scale_intervals[(prog_idx + 6) % len(scale_intervals)]
                chord_notes.append(note4)

            chord_dur = bar_duration_s * 1.1
            chord_len = min(int(sample_rate * chord_dur), num_samples - b_idx)
            if chord_len <= 0:
                continue

            t = np.linspace(0, chord_dur, chord_len, False)
            chord_sig = np.zeros(chord_len, dtype=np.float32)

            for note in chord_notes:
                f0 = midi_to_freq(note)
                if harmony_type in ("supersaw_pad", "orchestral_strings"):
                    # Unison detuning for lush pad
                    osc1 = np.sin(2.0 * np.pi * f0 * t)
                    osc2 = np.sin(2.0 * np.pi * (f0 * 1.003) * t)
                    osc3 = np.sin(2.0 * np.pi * (f0 * 0.997) * t)
                    note_sig = (osc1 + 0.8 * osc2 + 0.8 * osc3) / 2.6
                elif harmony_type == "acoustic_strum":
                    # Pluck with harmonics
                    note_sig = np.sin(2.0 * np.pi * f0 * t) + 0.5 * np.sin(4.0 * np.pi * f0 * t) + 0.25 * np.sin(6.0 * np.pi * f0 * t)
                else:
                    # Warm smooth sine pad
                    note_sig = np.sin(2.0 * np.pi * f0 * t) + 0.3 * np.sin(4.0 * np.pi * f0 * t)

                chord_sig += note_sig.astype(np.float32)

            chord_sig /= len(chord_notes)

            # Filter & Envelope
            sos = signal.butter(2, 2800, btype="lowpass", fs=sample_rate, output="sos")
            chord_filtered = signal.sosfilt(sos, chord_sig)

            # Smooth ADSR: attack 60ms, release tail
            env = np.ones(chord_len, dtype=np.float32)
            attack_len = min(int(sample_rate * 0.08), chord_len)
            env[:attack_len] = np.linspace(0.0, 1.0, attack_len)
            # Gentle exponential decay
            env *= np.exp(-t * 0.4).astype(np.float32)

            harmony_audio[b_idx : b_idx + chord_len] += (chord_filtered * env * 0.65).astype(np.float32)

        return np.clip(harmony_audio, -1.0, 1.0)

    def _generate_melody_stem(
        self,
        num_samples: int,
        sample_rate: int,
        tempo_bpm: int,
        beat_times: List[float],
        root_midi: int,
        scale: str,
        progression: List[int],
        melody_style: str,
        sections_info: List[Dict[str, Any]],
        rng: np.random.RandomState,
    ) -> np.ndarray:
        melody_audio = np.zeros(num_samples, dtype=np.float32)
        beat_duration_s = 60.0 / tempo_bpm
        scale_intervals = SCALES.get(scale, SCALES["major"])
        melody_root = (root_midi % 12) + 72 # Octave 5

        for beat_num, b_time in enumerate(beat_times):
            b_idx = int(b_time * sample_rate)
            if b_idx >= num_samples:
                break

            # Section energy dictates melody presence
            sec_energy = 0.7
            for sec in sections_info:
                if b_time >= sec["effective_start_s"]:
                    sec_energy = sec["energy"]

            # Phrasing: play on some beats with musical rests
            if rng.random() > (0.4 + 0.4 * sec_energy):
                continue # Musical rest

            bar_idx = beat_num // 4
            prog_idx = progression[bar_idx % len(progression)]
            # Pick a melodic note from chord tone or passing pentatonic note
            step_offset = rng.choice([0, 2, 4, 7, 9, 11])
            note_midi = melody_root + scale_intervals[step_offset % len(scale_intervals)]
            freq = midi_to_freq(note_midi)

            note_dur = beat_duration_s * rng.choice([0.8, 1.2, 1.8])
            note_len = min(int(sample_rate * note_dur), num_samples - b_idx)
            if note_len <= 0:
                continue

            t = np.linspace(0, note_dur, note_len, False)
            # Gentle vibrato LFO (5 Hz)
            vibrato = np.sin(2.0 * np.pi * 5.0 * t) * (0.008 * freq)
            inst_freq = freq + vibrato
            phase = 2.0 * np.pi * np.cumsum(inst_freq) / sample_rate

            # Warm acoustic/synth lead
            lead = np.sin(phase) + 0.3 * np.sin(2.0 * phase) + 0.1 * np.sin(3.0 * phase)
            
            # Envelope
            attack_len = min(int(sample_rate * 0.04), note_len)
            env = np.ones(note_len, dtype=np.float32)
            env[:attack_len] = np.linspace(0.0, 1.0, attack_len)
            env *= np.exp(-t * 1.5).astype(np.float32)

            melody_audio[b_idx : b_idx + note_len] += (lead * env * 0.6 * sec_energy).astype(np.float32)

        return np.clip(melody_audio, -1.0, 1.0)

    def generate(self, request: MusicJobRequest, sample_rate: int = 48000) -> GenerationResult:
        # 1. Analyze prompt & check safety/warnings
        profile, warnings = self._analyze_prompt(request.prompt)

        # 2. Establish seed
        seed = (
            int(request.seed) % (2**32 - 5)
            if request.seed is not None
            else int(np.random.randint(0, 1_000_000))
        )
        rng = np.random.RandomState(seed)

        # 3. Tempo & Key determination
        tempo_bpm = request.tempo_bpm if request.tempo_bpm is not None else profile["bpm"]
        tempo_bpm = max(40, min(220, tempo_bpm))
        
        key_str = request.key if request.key else profile["key"]
        # Parse root and scale
        root_name = key_str.split()[0]
        root_midi = ROOT_NOTES.get(root_name, profile["root"])
        scale_name = profile["scale"]
        if "minor" in key_str.lower():
            scale_name = "minor"
        elif "major" in key_str.lower():
            scale_name = "major"

        beat_duration_s = 60.0 / tempo_bpm
        num_samples = int(round(request.duration_s * sample_rate))

        # 4. Generate Beat and Bar Grid
        total_beats = int(math.ceil(request.duration_s / beat_duration_s)) + 1
        beat_times = [round(i * beat_duration_s, 4) for i in range(total_beats) if i * beat_duration_s < request.duration_s]
        bar_times = [round(i * 4 * beat_duration_s, 4) for i in range(math.ceil(len(beat_times) / 4.0)) if i * 4 * beat_duration_s < request.duration_s]

        # 5. Process Sections & Align to Beat Grid (Advanced Requirement)
        sections_info = []
        if request.sections:
            sorted_sections = sorted(request.sections, key=lambda s: s.start_s)
            prev_start = -10.0
            for sec in sorted_sections:
                if sec.start_s >= request.duration_s:
                    warnings.append(f"Section '{sec.mood}' start_s ({sec.start_s}s) exceeds track duration ({request.duration_s}s); ignored.")
                    continue
                if sec.start_s - prev_start < 1.5:
                    warnings.append(f"Section '{sec.mood}' at {sec.start_s}s is under 1.5s from previous; spaced onto musical beat.")

                # Quantize to nearest beat within 0.5s
                nearest_beat = round(sec.start_s / beat_duration_s) * beat_duration_s
                if abs(nearest_beat - sec.start_s) > 0.5:
                    nearest_beat = sec.start_s # Fallback

                # Determine section energy based on mood
                sec_mood_lower = sec.mood.lower()
                sec_energy = 0.7
                if any(k in sec_mood_lower for k in ["calm", "intro", "soft", "ambient", "quiet"]):
                    sec_energy = 0.4
                elif any(k in sec_mood_lower for k in ["build", "rise", "transition"]):
                    sec_energy = 0.75
                elif any(k in sec_mood_lower for k in ["peak", "climax", "uplifting", "intense", "exciting", "heavy"]):
                    sec_energy = 1.0
                elif any(k in sec_mood_lower for k in ["outro", "ending", "fade"]):
                    sec_energy = 0.5

                sections_info.append({
                    "requested_start_s": sec.start_s,
                    "effective_start_s": round(nearest_beat, 4),
                    "mood": sec.mood,
                    "energy": sec_energy
                })
                prev_start = sec.start_s

        if not sections_info:
            sections_info = [{
                "requested_start_s": 0.0,
                "effective_start_s": 0.0,
                "mood": profile.get("detected_mood", "balanced"),
                "energy": profile["energy"]
            }]

        progression = profile["progressions"][0]

        # 6. Generate Real Independent Stems
        drums = self._generate_drum_stem(
            num_samples=num_samples,
            sample_rate=sample_rate,
            tempo_bpm=tempo_bpm,
            beat_times=beat_times,
            drum_density=profile["drum_density"],
            sections_info=sections_info,
            rng=np.random.RandomState(seed + 1),
            loopable=bool(request.loopable)
        )

        bass = self._generate_bass_stem(
            num_samples=num_samples,
            sample_rate=sample_rate,
            tempo_bpm=tempo_bpm,
            beat_times=beat_times,
            root_midi=root_midi,
            scale=scale_name,
            progression=progression,
            bass_type=profile["bass_type"],
            sections_info=sections_info,
            rng=np.random.RandomState(seed + 2)
        )

        harmony = self._generate_harmony_stem(
            num_samples=num_samples,
            sample_rate=sample_rate,
            tempo_bpm=tempo_bpm,
            beat_times=beat_times,
            root_midi=root_midi,
            scale=scale_name,
            progression=progression,
            harmony_type=profile["harmony_type"],
            sections_info=sections_info,
            rng=np.random.RandomState(seed + 3)
        )

        melody = self._generate_melody_stem(
            num_samples=num_samples,
            sample_rate=sample_rate,
            tempo_bpm=tempo_bpm,
            beat_times=beat_times,
            root_midi=root_midi,
            scale=scale_name,
            progression=progression,
            melody_style=profile["melody_style"],
            sections_info=sections_info,
            rng=np.random.RandomState(seed + 4)
        )

        # 7. Ending fade / musical termination or looping
        if request.loopable:
            # Crossfade 10ms at edges to guarantee seamless loop with zero click
            fade_len = int(sample_rate * 0.01)
            fade_curve = np.linspace(0.0, 1.0, fade_len)
            for stem in [drums, bass, harmony, melody]:
                stem[:fade_len] *= fade_curve
                stem[-fade_len:] *= fade_curve[::-1]
        else:
            # Musical ending: natural release into quiet in the final 2.0s
            ending_dur_s = min(2.5, request.duration_s * 0.25)
            ending_samples = int(sample_rate * ending_dur_s)
            ending_fade = np.linspace(1.0, 0.0, ending_samples) ** 1.5
            for stem in [drums, bass, harmony, melody]:
                stem[-ending_samples:] *= ending_fade

        stems = {
            "drums": drums,
            "bass": bass,
            "harmony": harmony,
            "melody": melody
        }

        stem_gains = {
            "drums": 0.80,
            "bass": 0.75,
            "harmony": 0.65,
            "melody": 0.70
        }

        generation_method_info = {
            "provider": "ProceduralSynthesizerProvider",
            "method": "Rule-based procedural DSP synthesis (not a trained AI music model)",
            "version": "1.0.0",
            "audio_synthesis": "Additive & Subtractive Harmonic DSP with State-Variable Filters",
            "seed": seed,
            "sections_aligned": sections_info
        }

        return GenerationResult(
            stems=stems,
            stem_gains=stem_gains,
            tempo_bpm=tempo_bpm,
            key=key_str,
            time_signature="4/4",
            beat_times=beat_times,
            bar_times=bar_times,
            generation_method_info=generation_method_info,
            warnings=warnings
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
        profile, _ = self._analyze_prompt(request.prompt)
        rng = np.random.RandomState(new_seed)
        num_samples = int(round(request.duration_s * sample_rate))
        beat_duration_s = 60.0 / tempo_bpm
        total_beats = int(math.ceil(request.duration_s / beat_duration_s)) + 1
        beat_times = [round(i * beat_duration_s, 4) for i in range(total_beats) if i * beat_duration_s < request.duration_s]
        
        root_name = key.split()[0]
        root_midi = ROOT_NOTES.get(root_name, profile["root"])
        scale_name = profile["scale"]
        progression = profile["progressions"][0]
        sections_info = [{
            "requested_start_s": 0.0,
            "effective_start_s": 0.0,
            "mood": profile.get("detected_mood", "balanced"),
            "energy": profile["energy"]
        }]

        if stem_name == "drums":
            stem = self._generate_drum_stem(
                num_samples, sample_rate, tempo_bpm, beat_times,
                profile["drum_density"], sections_info, rng, bool(request.loopable)
            )
        elif stem_name == "bass":
            stem = self._generate_bass_stem(
                num_samples, sample_rate, tempo_bpm, beat_times,
                root_midi, scale_name, progression, profile["bass_type"], sections_info, rng
            )
        elif stem_name == "harmony":
            stem = self._generate_harmony_stem(
                num_samples, sample_rate, tempo_bpm, beat_times,
                root_midi, scale_name, progression, profile["harmony_type"], sections_info, rng
            )
        elif stem_name == "melody":
            stem = self._generate_melody_stem(
                num_samples, sample_rate, tempo_bpm, beat_times,
                root_midi, scale_name, progression, profile["melody_style"], sections_info, rng
            )
        else:
            raise ValueError(f"Unknown stem name: {stem_name}")

        if not request.loopable:
            ending_dur_s = min(2.5, request.duration_s * 0.25)
            ending_samples = int(sample_rate * ending_dur_s)
            ending_fade = np.linspace(1.0, 0.0, ending_samples) ** 1.5
            stem[-ending_samples:] *= ending_fade

        return stem
