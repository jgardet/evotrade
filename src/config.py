from __future__ import annotations
from datetime import date, timedelta
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from pydantic import field_validator, model_validator
from typing import Annotated, Literal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        enable_decoding=False,
    )

    # LLM
    llm_backend: Literal["claude", "codex"] = "claude"
    anthropic_api_key: str = ""
    claude_model: str = "claude-opus-4-6"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    # Database
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_db: str = "freqtrade_agent"
    postgres_user: str = "agent"
    postgres_password: str = "changeme"

    # Redis
    redis_url: str = "redis://redis:6379/0"

    # Freqtrade paths
    freqtrade_config: str = "/freqtrade/config.json"
    freqtrade_data_dir: str = "/freqtrade/user_data/data"
    freqtrade_strategy_dir: str = "/freqtrade/user_data/strategies"
    freqtrade_results_dir: str = "/freqtrade/user_data/backtest_results"

    # ----------------------------------------------------------------
    # Backtest universe — what the agent is allowed to test
    # ----------------------------------------------------------------
    # Comma-separated in .env:  BACKTEST_PAIRS=BTC/USDT,ETH/USDT
    backtest_pairs: Annotated[list[str], NoDecode] = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ADA/USDT"]
    backtest_timeframes: Annotated[list[str], NoDecode] = ["1h", "4h", "1d"]

    # ----------------------------------------------------------------
    # Date windows — set once, never change after first run
    # ----------------------------------------------------------------
    in_sample_start: date = date(2021, 1, 1)
    in_sample_end: date = date(2023, 6, 30)
    holdout_start: date = date(2023, 7, 1)
    holdout_end: date = date(2024, 6, 30)

    # ----------------------------------------------------------------
    # Walk-forward validation
    # ----------------------------------------------------------------
    walk_forward_enabled: bool = True
    # Size of each rolling in-sample window (days)
    walk_forward_window_days: int = 180
    # How far each window shifts forward (days)
    walk_forward_step_days: int = 60
    # Minimum windows a strategy must pass to be considered robust
    walk_forward_min_pass_ratio: float = 0.6

    # ----------------------------------------------------------------
    # Agent thresholds
    # ----------------------------------------------------------------
    max_drawdown_threshold: float = 0.25
    min_trade_count: int = 30
    oos_holdout_ratio: float = 0.2
    insight_synthesis_every: int = 20
    max_components_per_strategy: int = 6

    log_level: str = "INFO"

    # ----------------------------------------------------------------
    # Validators
    # ----------------------------------------------------------------

    @field_validator("backtest_pairs", "backtest_timeframes", mode="before")
    @classmethod
    def parse_csv(cls, v):
        """Allow comma-separated strings from env vars."""
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        return v

    @model_validator(mode="after")
    def validate_date_windows(self) -> "Settings":
        if self.in_sample_end >= self.holdout_start:
            raise ValueError(
                f"in_sample_end ({self.in_sample_end}) must be before "
                f"holdout_start ({self.holdout_start})"
            )
        if self.holdout_start >= self.holdout_end:
            raise ValueError("holdout_start must be before holdout_end")
        if not self.backtest_pairs:
            raise ValueError("backtest_pairs must not be empty")
        if not self.backtest_timeframes:
            raise ValueError("backtest_timeframes must not be empty")
        return self

    # ----------------------------------------------------------------
    # Derived properties
    # ----------------------------------------------------------------

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def async_postgres_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def available_contexts(self) -> list[dict]:
        """Full cross-product of configured pairs × timeframes."""
        return [
            {"pair": p, "timeframe": tf}
            for p in self.backtest_pairs
            for tf in self.backtest_timeframes
        ]

    @property
    def walk_forward_windows(self) -> list[tuple[date, date]]:
        """
        Generate rolling in-sample windows across the in-sample period.
        Each window is walk_forward_window_days wide and shifts by
        walk_forward_step_days. The holdout period is never touched.
        """
        windows: list[tuple[date, date]] = []
        window_size = timedelta(days=self.walk_forward_window_days)
        step = timedelta(days=self.walk_forward_step_days)
        start = self.in_sample_start

        while start + window_size <= self.in_sample_end:
            windows.append((start, start + window_size))
            start += step

        return windows

settings = Settings()
