from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class MusicSection(BaseModel):
    start_s: float = Field(..., description="Start time of the section in seconds", ge=0.0)
    mood: str = Field(..., description="Mood or description for this section")


class OutputOptions(BaseModel):
    sample_rate: int = Field(default=48000, description="Audio sample rate in Hz; output is fixed at 48000")
    target_lufs: float = Field(default=-16.0, description="Integrated loudness target in LUFS (default -16)")
    formats: List[str] = Field(default_factory=lambda: ["wav", "mp3"], description="List of audio output formats")


class MusicJobRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Text description of desired music mood/style")
    duration_s: float = Field(..., description="Duration in seconds (between 5 and 300)")
    sections: Optional[List[MusicSection]] = Field(default=None, description="Optional list of section mood changes")
    tempo_bpm: Optional[int] = Field(default=None, description="Optional tempo in BPM (40-220)")
    key: Optional[str] = Field(default=None, description="Optional musical key (e.g. 'C Major', 'A Minor')")
    loopable: bool = Field(default=False, description="Whether track must seamlessly loop")
    narration_url: Optional[str] = Field(default=None, description="Optional URL or path to voiceover file for ducking")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible generation")
    output: Optional[OutputOptions] = Field(default_factory=OutputOptions, description="Audio format configuration")


class RegenerateStemRequest(BaseModel):
    stem_name: str = Field(..., description="Name of stem to regenerate (e.g., 'drums', 'bass', 'harmony', 'melody')")
    seed: Optional[int] = Field(default=None, description="New seed for regenerating this specific stem")


class ErrorDetails(BaseModel):
    code: str
    message: str
    retryable: bool
    details: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    error: ErrorDetails


class StemMetadata(BaseModel):
    filename: str
    download_url: str
    gain: float
    integrated_lufs: float
    peak_dbfs: float
    duration_s: float


class DuckingMetadata(BaseModel):
    voiceover_detected: bool
    ducking_reduction_db: float
    attack_ms: float
    release_ms: float
    curve_sample_points: List[float] = Field(default_factory=list)


class MusicManifest(BaseModel):
    job_id: str
    prompt: str
    duration_s: float
    delivered_duration_s: float
    tempo_bpm: int
    key: str
    time_signature: str = "4/4"
    seed: int
    sample_rate: int
    target_lufs: float
    actual_lufs: float
    peak_dbfs: float
    loopable: bool
    sections: List[Dict[str, Any]]
    beat_grid: Dict[str, Any]
    stems: Dict[str, StemMetadata]
    generation_method: Dict[str, Any]
    timings: Dict[str, Any]
    warnings: List[str] = Field(default_factory=list)
    ducking: Optional[DuckingMetadata] = None
    licensing: Dict[str, Any]


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: int = Field(ge=0, le=100)
    created_at: str
    updated_at: str
    request: Optional[MusicJobRequest] = None
    downloads: Optional[Dict[str, str]] = None
    manifest: Optional[Dict[str, Any]] = None
    error: Optional[ErrorDetails] = None
