"""Retry file writes that fail because the disk is full (ENOSPC / EDQUOT).

A shared PVC hitting 100% (or a per-user/-project quota) makes a ``write()`` /
``flush()`` / ``fsync()`` raise ``OSError [Errno 28]`` (ENOSPC) or ``[Errno 122]``
(EDQUOT). Un-caught, that kills the whole sim mid-day and — because ``--resume``
is day-granular — discards hours of in-progress work. But the condition is almost
always TRANSIENT: on a shared cluster other tenants free space within minutes.

So we treat a full disk like a blocking storage hang: log once, sleep with capped
backoff, and retry the SAME operation until it succeeds. The process stays alive
and continues from the exact session once space returns — zero loss, no resume
needed. Any non-disk-full ``OSError`` propagates unchanged.

Caller contract: the retried callable MUST be idempotent under a disk-full retry.
Pass a ``flush()`` (re-flushing the same buffer just re-attempts the pending
bytes) or a tmp-file + ``os.replace`` atomic write — NOT a bare ``write(line)``,
which would duplicate the line on retry. The write-then-retry-flush pattern in
the call sites keeps ``write()`` (buffer append) exactly-once and only retries the
disk-touching ``flush()``.
"""
from __future__ import annotations

import errno
import logging
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

# ENOSPC (28) = filesystem full; EDQUOT (122 on Linux) = per-user/-project quota
# exceeded. Both are what a shared PVC at 100% surfaces on write/flush/fsync.
# Hardcode 122 (the Linux EDQUOT) rather than trust errno.EDQUOT, which on
# some platforms (Windows) is a different Winsock code — the runtime target is
# Linux, and a real EDQUOT there raises errno 122.
_DISK_FULL_ERRNOS = frozenset(
    {errno.ENOSPC, 122} | ({errno.EDQUOT} if hasattr(errno, "EDQUOT") else set())
)

_T = TypeVar("_T")


def is_disk_full(exc: BaseException) -> bool:
    """True iff `exc` is a disk-full / quota-exceeded OSError."""
    return isinstance(exc, OSError) and exc.errno in _DISK_FULL_ERRNOS


def retry_on_disk_full(
    fn: Callable[[], _T],
    *,
    what: str = "write",
    initial_sleep: float = 1.0,
    max_sleep: float = 30.0,
) -> _T:
    """Call ``fn()``, retrying forever on a disk-full ``OSError``; re-raise anything
    else. Logs a single WARNING when it first stalls and a single INFO on recovery.

    NOTE: this blocks the calling thread while stalled. That is the intended
    behaviour — a full disk means no session can make durable progress anyway, so
    pausing here (like a kernel-level storage hang would) until space frees is
    exactly the zero-loss "live-process stall" we want.
    """
    sleep = initial_sleep
    stalled = False
    while True:
        try:
            result = fn()
            if stalled:
                logger.info("disk-full stall on %s RECOVERED; writes resumed.", what)
            return result
        except OSError as exc:
            if exc.errno not in _DISK_FULL_ERRNOS:
                raise
            if not stalled:
                logger.warning(
                    "disk full during %s (errno %s: %s) — STALLING, retrying until "
                    "space frees. Process alive, no data lost.",
                    what, exc.errno, exc.strerror,
                )
                stalled = True
            time.sleep(sleep)
            sleep = min(sleep * 2.0, max_sleep)
