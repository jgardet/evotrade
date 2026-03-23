"""
Backtest Runner — wraps the Freqtrade CLI to execute backtests
and parses the JSON output into structured BacktestRun objects.
"""
from __future__ import annotations
import asyncio
import json
import math
import os
import uuid
import zipfile
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import structlog

from src.config import settings
from src.models import (
    BacktestMetrics, BacktestRun, RegimeTrend, RegimeVol, Strategy,
)

log = structlog.get_logger()


class BacktestRunner:
    """
    Executes Freqtrade backtests via subprocess.
    Handles strategy file writing, CLI invocation, and result parsing.
    """

    def __init__(self):
        self.strategy_dir = Path(settings.freqtrade_strategy_dir)
        self.results_dir = Path(settings.freqtrade_results_dir)
        self.data_dir = Path(settings.freqtrade_data_dir)

    async def run(
        self,
        strategy: Strategy,
        pair: str,
        timeframe: str,
        date_from: date,
        date_to: date,
        is_holdout: bool = False,
    ) -> BacktestRun:
        """Write strategy to disk, run backtest, parse results."""
        strategy_file = self._write_strategy(strategy)
        log.info(
            "backtest.starting",
            strategy=strategy.name,
            pair=pair,
            timeframe=timeframe,
            date_from=str(date_from),
            date_to=str(date_to),
            is_holdout=is_holdout,
        )

        try:
            raw = await self._execute(strategy, pair, timeframe, date_from, date_to)
        except BacktestError as e:
            if self._is_missing_data_error(str(e)):
                log.info(
                    "backtest.missing_data.detected",
                    pair=pair,
                    timeframe=timeframe,
                    date_from=str(date_from),
                    date_to=str(date_to),
                )
                try:
                    await self._download_data(pair, timeframe, date_from, date_to)
                    raw = await self._execute(strategy, pair, timeframe, date_from, date_to)
                except BacktestError as retry_error:
                    log.error("backtest.failed_after_download", strategy=strategy.name, error=str(retry_error))
                    return self._make_failed_run(
                        strategy, pair, timeframe, date_from, date_to, is_holdout, str(retry_error)
                    )
            else:
                log.error("backtest.failed", strategy=strategy.name, error=str(e))
                return self._make_failed_run(strategy, pair, timeframe, date_from, date_to, is_holdout, str(e))
        finally:
            # Clean up strategy file after run
            if strategy_file.exists():
                strategy_file.unlink()

        metrics = self._parse_metrics(raw, strategy.name)
        regime_trend, regime_vol, regime_meta = await self._detect_regime(
            pair, timeframe, date_from, date_to
        )

        run_id = uuid.uuid4()
        oos_delta = None  # set externally when we have both in-sample and holdout runs
        stability = self._compute_trade_stability(raw, strategy.name)
        score, disqualified, reason = self._compute_composite_score(
            metrics,
            stability,
            strategy_name=strategy.name,
        )

        return BacktestRun(
            id=run_id,
            strategy_id=strategy.id,
            pair=pair,
            timeframe=timeframe,
            date_from=date_from,
            date_to=date_to,
            is_holdout=is_holdout,
            regime_trend=regime_trend,
            regime_vol=regime_vol,
            regime_metadata=regime_meta,
            metrics=metrics,
            oos_sharpe_delta=oos_delta,
            trade_count_stability=stability,
            composite_score=score,
            disqualified=disqualified,
            disqualification_reason=reason,
            raw_output=raw,
        )

    def _write_strategy(self, strategy: Strategy) -> Path:
        self.strategy_dir.mkdir(parents=True, exist_ok=True)
        path = self.strategy_dir / f"{strategy.name}.py"
        path.write_text(strategy.code)
        return path

    async def _execute(
        self,
        strategy: Strategy,
        pair: str,
        timeframe: str,
        date_from: date,
        date_to: date,
    ) -> dict[str, Any]:
        """Run freqtrade backtest CLI and return parsed JSON output."""
        results_file = self.results_dir / f"{strategy.name}_{uuid.uuid4().hex[:8]}.json"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "freqtrade", "backtesting",
            "--userdir", str(self.strategy_dir.parent),
            "--config", settings.freqtrade_config,
            "--strategy", strategy.name,
            "--strategy-path", str(self.strategy_dir),
            "--pairs", pair,
            "--timeframe", timeframe,
            "--timerange", f"{date_from.strftime('%Y%m%d')}-{date_to.strftime('%Y%m%d')}",
            "--export", "trades",
            "--export-filename", str(results_file),
            "--cache", "none",
        ]

        log.debug("backtest.cmd", cmd=" ".join(cmd))

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=300  # 5 min max per backtest
        )

        if process.returncode != 0:
            raise BacktestError(
                f"Freqtrade exited with code {process.returncode}: "
                f"{stderr.decode()[-500:]}"
            )

        # Read the exported results JSON
        if results_file.exists():
            raw = json.loads(results_file.read_text())
            results_file.unlink()  # cleanup
            return raw

        artifact_raw = self._load_backtest_artifact(results_file)
        if artifact_raw is not None:
            return artifact_raw

        raise BacktestError("Freqtrade produced no results file")

    def _load_backtest_artifact(self, expected_results_file: Path) -> Optional[dict[str, Any]]:
        """
        Fallback loader for newer Freqtrade output formats.
        Tries .last_result.json and then latest zip/json artifacts under results_dir.
        """
        artifact_path = self._resolve_latest_artifact()
        if artifact_path is None:
            log.warning(
                "backtest.results_missing",
                expected_file=str(expected_results_file),
                results_dir=str(self.results_dir),
            )
            return None

        try:
            raw = self._read_artifact_payload(artifact_path)
            log.info(
                "backtest.results_fallback_used",
                expected_file=str(expected_results_file),
                artifact_file=str(artifact_path),
            )
            return raw
        except Exception as e:
            log.warning(
                "backtest.results_fallback_failed",
                artifact_file=str(artifact_path),
                error=str(e),
            )
            return None

    def _resolve_latest_artifact(self) -> Optional[Path]:
        last_result_file = self.results_dir / ".last_result.json"
        if last_result_file.exists():
            try:
                data = json.loads(last_result_file.read_text())
                latest = data.get("latest_backtest")
                if latest:
                    artifact = self.results_dir / latest
                    if artifact.exists():
                        return artifact
            except Exception:
                pass

        candidates = sorted(
            [p for p in self.results_dir.glob("backtest-result-*") if p.is_file()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    def _read_artifact_payload(self, artifact_path: Path) -> dict[str, Any]:
        if artifact_path.suffix.lower() == ".zip":
            with zipfile.ZipFile(artifact_path, "r") as zf:
                json_names = [
                    n for n in zf.namelist()
                    if n.endswith(".json")
                    and not n.endswith("_config.json")
                    and "strategy" not in n.lower()
                ]
                if not json_names:
                    json_names = [n for n in zf.namelist() if n.endswith(".json")]
                if not json_names:
                    raise BacktestError(f"Zip artifact has no JSON payload: {artifact_path.name}")
                return json.loads(zf.read(json_names[0]))

        if artifact_path.suffix.lower() == ".json":
            return json.loads(artifact_path.read_text())

        raise BacktestError(f"Unsupported artifact type: {artifact_path.name}")

    def _is_missing_data_error(self, error_message: str) -> bool:
        msg = error_message.lower()
        return (
            "no data found" in msg
            or "no history for" in msg
            or "download-data" in msg
        )

    async def _download_data(
        self,
        pair: str,
        timeframe: str,
        date_from: date,
        date_to: date,
    ) -> None:
        timerange = f"{date_from.strftime('%Y%m%d')}-{date_to.strftime('%Y%m%d')}"
        cmd = [
            "freqtrade", "download-data",
            "--userdir", str(self.strategy_dir.parent),
            "--config", settings.freqtrade_config,
            "--pairs", pair,
            "--timeframes", timeframe,
            "--timerange", timerange,
        ]
        log.info(
            "backtest.missing_data.downloading",
            pair=pair,
            timeframe=timeframe,
            timerange=timerange,
        )

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=900)

        if process.returncode != 0:
            raise BacktestError(
                f"Download-data exited with code {process.returncode}: "
                f"{stderr.decode()[-500:]}"
            )

        log.info(
            "backtest.missing_data.download_complete",
            pair=pair,
            timeframe=timeframe,
            output_tail=stdout.decode()[-300:],
        )

    def _parse_metrics(self, raw: dict, strategy_name: str) -> BacktestMetrics:
        """Extract metrics from freqtrade's backtest result JSON."""
        try:
            strat = raw.get("strategy", {}).get(strategy_name, {})
            results = strat.get("results_per_pair", [{}])
            total = strat.get("results") or strat

            if not isinstance(total, dict):
                total = {}

            trade_count = total.get("total_trades", total.get("trades", 0)) or 0
            profit_pct = total.get("profit_percent", None)
            if profit_pct is None:
                profit_pct = total.get("profit_total_pct", None)

            max_dd_raw = (
                total.get("max_drawdown")
                or total.get("max_drawdown_account")
                or total.get("max_relative_drawdown")
                or 0
            )
            max_dd = abs(max_dd_raw)
            win_rate = total.get("wins", 0) / max(trade_count, 1)

            # Monthly breakdown for stability calc
            monthly = strat.get("periodic_breakdown", {}).get("monthly", [])
            monthly_profits = [m.get("profit_percent", 0) for m in monthly]
            monthly_mean = float(np.mean(monthly_profits)) if monthly_profits else None
            monthly_std = float(np.std(monthly_profits)) if monthly_profits else None

            # Freqtrade doesn't compute Sharpe natively; we derive it
            sharpe = self._compute_sharpe(monthly_profits)
            sortino = self._compute_sortino(monthly_profits)
            calmar = profit_pct / max(max_dd, 0.001) if profit_pct and max_dd else None
            profit_factor = total.get("profit_factor", None)

            holding_avg_s = total.get("holding_avg_s")
            avg_dur = (holding_avg_s / 3600) if holding_avg_s else None  # seconds → hours

            return BacktestMetrics(
                sharpe=sharpe,
                sortino=sortino,
                profit_factor=profit_factor,
                max_drawdown=max_dd,
                win_rate=win_rate,
                trade_count=trade_count,
                avg_trade_dur_h=avg_dur,
                calmar=calmar,
                total_profit_pct=profit_pct,
                monthly_profit_mean=monthly_mean,
                monthly_profit_std=monthly_std,
            )
        except Exception as e:
            log.warning("backtest.parse_error", error=str(e))
            return BacktestMetrics(
                sharpe=None, sortino=None, profit_factor=None, max_drawdown=None,
                win_rate=None, trade_count=0, avg_trade_dur_h=None, calmar=None,
                total_profit_pct=None, monthly_profit_mean=None, monthly_profit_std=None,
            )

    def _compute_sharpe(self, monthly_returns: list[float], rf: float = 0.0) -> Optional[float]:
        if len(monthly_returns) < 3:
            return None
        arr = np.array(monthly_returns)
        excess = arr - rf / 12
        std = np.std(excess)
        if std == 0:
            return None
        return float(np.mean(excess) / std * np.sqrt(12))

    def _compute_sortino(self, monthly_returns: list[float], rf: float = 0.0) -> Optional[float]:
        if len(monthly_returns) < 3:
            return None
        arr = np.array(monthly_returns)
        excess = arr - rf / 12
        downside = arr[arr < 0]
        downside_std = np.std(downside) if len(downside) > 0 else 0
        if downside_std == 0:
            return None
        return float(np.mean(excess) / downside_std * np.sqrt(12))

    def _compute_trade_stability(self, raw: dict, strategy_name: str) -> Optional[float]:
        """Coefficient of variation of monthly trade counts. Lower = more stable."""
        try:
            strat = raw.get("strategy", {}).get(strategy_name, {})
            monthly = strat.get("periodic_breakdown", {}).get("monthly", [])
            counts = [m.get("trades", 0) for m in monthly]
            if len(counts) < 2:
                return None
            mean = np.mean(counts)
            if mean == 0:
                return 1.0
            return float(np.std(counts) / mean)
        except Exception:
            return None

    def _compute_composite_score(
        self,
        metrics: BacktestMetrics,
        stability: Optional[float],
        strategy_name: str = "",
    ) -> tuple[Optional[float], bool, Optional[str]]:
        """
        Returns (score, disqualified, reason).
        See the scoring model in the architecture docs.
        """
        # Hard disqualifiers
        if (metrics.trade_count or 0) < settings.min_trade_count:
            return None, True, f"Insufficient trades: {metrics.trade_count} < {settings.min_trade_count}"
        drawdown_threshold = settings.max_drawdown_threshold
        if self._is_explore_strategy(strategy_name):
            drawdown_threshold = settings.explore_max_drawdown_threshold

        if metrics.max_drawdown and metrics.max_drawdown > drawdown_threshold:
            return None, True, f"Max drawdown too high: {metrics.max_drawdown:.1%}"
        if metrics.sharpe is None:
            return None, True, "Could not compute Sharpe ratio"

        # Base score
        sharpe = metrics.sharpe or 0
        sortino = metrics.sortino or sharpe
        calmar = min(metrics.calmar or 0, 5.0)  # cap at 5 to avoid outlier inflation
        base = sharpe * 0.5 + sortino * 0.3 + calmar * 0.2

        # Stability bonus
        stab_bonus = (1 - min(stability or 1.0, 1.0)) * 0.2

        # Complexity penalty (applied externally when full strategy is known)
        score = base + stab_bonus

        return round(score, 4), False, None

    def _is_explore_strategy(self, strategy_name: str) -> bool:
        return strategy_name.startswith("Agent_EXPLORE_")

    async def _detect_regime(
        self, pair: str, timeframe: str, date_from: date, date_to: date
    ) -> tuple[RegimeTrend, RegimeVol, dict]:
        """
        Simple regime detection using price data.
        In production, extend with ADX, Bollinger Band width, etc.
        """
        # Placeholder: in a real deployment, load OHLCV data and compute regime
        # This would query the freqtrade data directory for the pair's candles
        return RegimeTrend.UNKNOWN, RegimeVol.UNKNOWN, {}

    def _make_failed_run(
        self, strategy, pair, timeframe, date_from, date_to, is_holdout, reason
    ) -> BacktestRun:
        return BacktestRun(
            id=uuid.uuid4(),
            strategy_id=strategy.id,
            pair=pair,
            timeframe=timeframe,
            date_from=date_from,
            date_to=date_to,
            is_holdout=is_holdout,
            regime_trend=RegimeTrend.UNKNOWN,
            regime_vol=RegimeVol.UNKNOWN,
            regime_metadata={},
            metrics=BacktestMetrics(
                sharpe=None, sortino=None, profit_factor=None, max_drawdown=None,
                win_rate=None, trade_count=0, avg_trade_dur_h=None, calmar=None,
                total_profit_pct=None, monthly_profit_mean=None, monthly_profit_std=None,
            ),
            oos_sharpe_delta=None,
            trade_count_stability=None,
            composite_score=None,
            disqualified=True,
            disqualification_reason=reason,
            raw_output={"error": reason},
        )


class BacktestError(Exception):
    pass


runner = BacktestRunner()
