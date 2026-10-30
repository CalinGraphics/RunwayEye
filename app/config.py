from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Video source
    # File path, camera index ("0") or RTSP URL. Empty = no source (placeholder frames).
    video_source: str = "0"
    # Used automatically when VIDEO_SOURCE cannot be opened.
    fallback_video: str = "data/videos/fallback.mp4"
    camera_width: int = 1280
    camera_height: int = 720
    # All processing, overlays and calibration happen at this frame width.
    frame_width: int = 960
    camera_id: str = "cam-01"
    camera_name: str = "Camera 1 · machetă"

    # YOLO
    enable_yolo: bool = True
    yolo_model_path: str = "models/best.pt"
    # Provisional pretrained model, used only while best.pt does not exist.
    yolo_fallback_model: str = "models/yolo11n.pt"
    yolo_conf: float = 0.35
    yolo_imgsz: int = 640
    yolo_every_n: int = 2

    # Change detection
    reference_frames: int = 15
    diff_threshold: int = 30
    blur_ksize: int = 3
    min_area_px: int = 60
    max_area_px: int = 6000
    bg_learning_rate: float = 0.02
    # Fraction of the runway mask that changed at once => lighting change, not an object.
    global_change_frac: float = 0.45

    # Persistence / tracking
    persist_frames: int = 8
    large_persist_frames: int = 30
    gone_frames: int = 20
    stationary_px: int = 14
    match_px: int = 40
    dedup_px: int = 40

    # Risk
    unknown_critical_cm: float = 15.0

    # Runway (values to be verified with the airport)
    runway_name: str = "09/27"
    runway_length_m: float = 2500.0
    runway_width_m: float = 45.0
    sectors_along: int = 6
    sectors_across: int = 3
    # Physical size of the runway painted on the mock-up, in cm (0 = real runway).
    mockup_length_cm: float = 100.0
    mockup_width_cm: float = 60.0

    # Data
    use_local_data: bool = True
    enable_supabase: bool = False
    supabase_url: str = ""
    supabase_key: str = ""
    supabase_bucket: str = "fod-images"

    # Claude (bonus)
    enable_claude: bool = False
    anthropic_api_key: str = ""
    claude_model: str = "claude-opus-5-5"

    # App
    demo_mode: bool = True
    demo_fake_detections: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    # Key other devices need (it's in the QR code). Empty = generated once into data/access_key.txt.
    access_key: str = ""
    data_dir: str = "data"

    def path(self, value: str) -> Path:
        """Resolve a path setting relative to the project root."""
        p = Path(value)
        return p if p.is_absolute() else ROOT_DIR / p

    @property
    def data_path(self) -> Path:
        return self.path(self.data_dir)

    @property
    def calibration_file(self) -> Path:
        return self.data_path / "calibration.json"

    @property
    def tuning_file(self) -> Path:
        return self.data_path / "tuning.json"

    @property
    def supabase_active(self) -> bool:
        return (
            self.enable_supabase
            and not self.use_local_data
            and bool(self.supabase_url)
            and bool(self.supabase_key)
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
