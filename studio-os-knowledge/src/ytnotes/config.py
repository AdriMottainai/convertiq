"""Configuration loading and validation for ytnotes.

Loads ``config/channels.yaml`` and ``config/taxonomy.yaml`` into validated
pydantic models, and resolves runtime settings (model, per-run caps, API key)
from a mix of config defaults and environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

# Repo root is two levels up from this file: src/ytnotes/config.py -> repo/
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = REPO_ROOT / "config"
DEFAULT_STATE_PATH = REPO_ROOT / "state" / "processed.json"
DEFAULT_VAULT_DIR = REPO_ROOT / "vault"

DEFAULT_MODEL = "claude-sonnet-5"


# --------------------------------------------------------------------------- #
# Taxonomy
# --------------------------------------------------------------------------- #
class TaxonomyEntry(BaseModel):
    id: str
    description: str = ""


class Taxonomy(BaseModel):
    agents: list[TaxonomyEntry] = Field(default_factory=list)
    skills: list[TaxonomyEntry] = Field(default_factory=list)

    @field_validator("agents", "skills")
    @classmethod
    def _unique_ids(cls, v: list[TaxonomyEntry]) -> list[TaxonomyEntry]:
        ids = [e.id for e in v]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate taxonomy ids: {sorted(dupes)}")
        return v

    def agent_ids(self) -> list[str]:
        return [e.id for e in self.agents]

    def skill_ids(self) -> list[str]:
        return [e.id for e in self.skills]


# --------------------------------------------------------------------------- #
# Channels
# --------------------------------------------------------------------------- #
class ChannelDefaults(BaseModel):
    languages: list[str] = Field(default_factory=lambda: ["en"])
    whisper_fallback: bool = True
    first_seen_max_age_days: int = 14


class ChannelConfig(BaseModel):
    name: str
    channel_id: str | None = None
    handle: str | None = None
    url: str | None = None
    agents_hint: list[str] = Field(default_factory=list)
    skills_hint: list[str] = Field(default_factory=list)

    def has_target(self) -> bool:
        return bool(self.channel_id or self.handle or self.url)


class ChannelsFile(BaseModel):
    defaults: ChannelDefaults = Field(default_factory=ChannelDefaults)
    channels: list[ChannelConfig] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Runtime settings
# --------------------------------------------------------------------------- #
class Settings(BaseModel):
    """Everything the pipeline needs at runtime, assembled from files + env."""

    model: str = DEFAULT_MODEL
    max_videos_per_run: int = 25
    max_whisper_per_run: int = 3
    use_thinking: bool = False
    max_transcript_chars: int = 200_000  # guard against pathological outliers

    config_dir: Path = DEFAULT_CONFIG_DIR
    state_path: Path = DEFAULT_STATE_PATH
    vault_dir: Path = DEFAULT_VAULT_DIR

    channels: ChannelsFile = Field(default_factory=ChannelsFile)
    taxonomy: Taxonomy = Field(default_factory=Taxonomy)

    model_config = {"arbitrary_types_allowed": True}


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_settings(
    config_dir: Path | str = DEFAULT_CONFIG_DIR,
    *,
    state_path: Path | str = DEFAULT_STATE_PATH,
    vault_dir: Path | str = DEFAULT_VAULT_DIR,
) -> Settings:
    """Load channels + taxonomy and layer env overrides on top."""
    config_dir = Path(config_dir)
    channels = ChannelsFile.model_validate(_read_yaml(config_dir / "channels.yaml"))
    taxonomy = Taxonomy.model_validate(_read_yaml(config_dir / "taxonomy.yaml"))

    settings = Settings(
        model=os.getenv("YTNOTES_MODEL", DEFAULT_MODEL),
        max_videos_per_run=int(os.getenv("YTNOTES_MAX_VIDEOS_PER_RUN", "25")),
        max_whisper_per_run=int(os.getenv("YTNOTES_MAX_WHISPER_PER_RUN", "3")),
        use_thinking=os.getenv("YTNOTES_USE_THINKING", "").lower() in {"1", "true", "yes"},
        config_dir=config_dir,
        state_path=Path(state_path),
        vault_dir=Path(vault_dir),
        channels=channels,
        taxonomy=taxonomy,
    )
    return settings


def get_api_key() -> str:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to .env locally or as a GitHub "
            "Actions repository secret."
        )
    return key
