"""
ARQ worker — handles background tasks dispatched from the agent loop.
Allows experiments to run in parallel without blocking the main orchestrator.
"""
from arq import cron
from arq.connections import RedisSettings

from src.config import settings
from src.db.database import db
from src.agent.main import AgentLoop
from src.hypothesis.synthesizer import synthesizer

import structlog
log = structlog.get_logger()

agent = AgentLoop()


async def run_experiment(ctx, force_type: str = None):
    """Background task: run one experiment cycle."""
    await agent.run_once(force_type=force_type)


async def run_synthesis(ctx):
    """Background task: run insight synthesis."""
    await synthesizer.synthesize()


async def startup(ctx):
    await db.connect()
    log.info("worker.started")


async def shutdown(ctx):
    await db.disconnect()
    log.info("worker.stopped")


class WorkerSettings:
    functions = [run_experiment, run_synthesis]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    max_jobs = 2  # run 2 experiments in parallel
    job_timeout = 600  # 10 min max per experiment
    cron_jobs = [
        # Auto-trigger synthesis every hour
        cron(run_synthesis, hour={*range(24)}, minute=0)
    ]
