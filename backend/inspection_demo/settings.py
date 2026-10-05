from pathlib import Path
from tempfile import gettempdir

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INSPECTION_DEMO_",
        env_file=".env",
        extra="ignore",
    )

    data_dir: Path = Path(gettempdir()) / "inspection-demo" / "data"
    asset_mount: Path = Path("/mnt/default-assets")
    expected_manifest_path: Path = Path(__file__).with_name("expected_assets.json")
    runtime_workspace: Path = Path(gettempdir()) / "inspection-demo" / "runtime"
    simulation_event_period_seconds: float = Field(default=0.0, ge=0.0, le=1.0)
    research_device: str = "cuda"
    grounding_timeout_seconds: int = Field(default=600, ge=30, le=600)
    frontend_dist: Path = Path(__file__).resolve().parents[2] / "frontend" / "dist"
