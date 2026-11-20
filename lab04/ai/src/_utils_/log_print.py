from __future__ import annotations

import sys
import time
from datetime import datetime


_t0 = time.monotonic()


def reset_log_timer() -> None:
    global _t0
    _t0 = time.monotonic()


def log_print(msg: str) -> None:
    elapsed = time.monotonic() - _t0
    print(f"[{elapsed:8.1f}s] {msg}", flush=True)


def inverted_unix_prefix(when: datetime | None = None) -> int:
    when = when or datetime.now()
    return 9_999_999_999 - int(when.timestamp())


def tensorboard_run_dir(log_root: str, when: datetime) -> str:
    from pathlib import Path

    name = f"{inverted_unix_prefix(when):010d}_{when.strftime('%Y%m%d-%H%M%S')}"
    p = Path(log_root) / name
    p.mkdir(parents=True, exist_ok=True)
    return str(p)