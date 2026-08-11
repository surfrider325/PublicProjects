"""Built-in cron scheduler.

Runs in the foreground and executes sources on their configured schedules.
Concurrency is bounded by a thread pool; the Postgres advisory lock taken in
runner.run_source means you can run several scheduler processes for redundancy
without double-loading a source.
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone

from croniter import croniter

from .config import SourceConfig
from .runner import RunResult, run_source

log = logging.getLogger(__name__)


class Scheduler:
    def __init__(
        self,
        sources: list[SourceConfig],
        *,
        dsn: str | None = None,
        max_workers: int = 4,
        tick_seconds: float = 1.0,
    ) -> None:
        self.sources = [s for s in sources if s.enabled and s.schedule]
        self.dsn = dsn
        self.tick_seconds = tick_seconds
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="apiflow")
        self.stopping = threading.Event()
        self.next_run: dict[str, datetime] = {}
        self.in_flight: dict[str, Future] = {}

    def _schedule_next(self, source: SourceConfig, after: datetime) -> None:
        assert source.schedule
        self.next_run[source.name] = croniter(source.schedule, after).get_next(datetime)

    def prime(self) -> None:
        now = datetime.now(timezone.utc)
        for source in self.sources:
            self._schedule_next(source, now)
            log.info(
                "scheduled %s (%s) — next run %s",
                source.name, source.schedule, self.next_run[source.name].isoformat(),
            )
        if not self.sources:
            log.warning("no enabled sources have a schedule; nothing to do")

    def stop(self, *_: object) -> None:
        log.info("shutdown requested; finishing in-flight runs")
        self.stopping.set()

    def _launch(self, source: SourceConfig) -> None:
        pending = self.in_flight.get(source.name)
        if pending and not pending.done():
            log.warning("[%s] previous run still going; skipping this tick", source.name)
            return

        def task() -> RunResult:
            result = run_source(source, dsn=self.dsn)
            log.info(result.summary())
            return result

        self.in_flight[source.name] = self.executor.submit(task)

    def run_forever(self) -> None:
        self.prime()
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)

        while not self.stopping.is_set():
            now = datetime.now(timezone.utc)
            for source in self.sources:
                due = self.next_run.get(source.name)
                if due and now >= due:
                    log.info("[%s] due at %s", source.name, due.isoformat())
                    self._launch(source)
                    self._schedule_next(source, now)
            time.sleep(self.tick_seconds)

        self.executor.shutdown(wait=True)
        log.info("scheduler stopped")
