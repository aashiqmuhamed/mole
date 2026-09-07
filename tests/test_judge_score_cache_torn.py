"""A torn line must cost ONE entry, not the rest of the cache.

Regression for a silent 93.2% cache loss. `_ScoreCache.__init__` wrapped the whole read
loop in one try/except, so the first unparseable line aborted the load: on the real
43 MB cache that meant 5,310 of 77,713 keys (aborting at line 7344 of 121,225). Nothing
failed -- the eval just silently re-paid for 72,403 verdicts it already had. The torn
lines come from concurrent processes appending in buffered text mode, so the writer is
fixed too.
"""
from __future__ import annotations

import json

from mole.monitors.llm_judge.judge import _ScoreCache


def _line(k, s=0.5):
    return json.dumps({"k": k, "s": s, "r": "why", "t": 10}) + "\n"


def test_torn_line_costs_one_entry_not_the_file(tmp_path):
    p = tmp_path / "c.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        fh.write(_line("a"))
        fh.write('iltration, sabotage, privilege abuse, or saf", "t": 8376}\n')  # real torn line
        fh.write(_line("b"))
        fh.write(_line("c"))
    c = _ScoreCache(str(p))
    assert set(c.mem) == {"a", "b", "c"}, "entries after the torn line must survive"
    assert c.torn == 1


def test_undecodable_byte_does_not_kill_the_load(tmp_path):
    p = tmp_path / "c.jsonl"
    with p.open("wb") as fh:
        fh.write(_line("a").encode())
        fh.write(b'{"k": "\x80bad", "s": 0.1}\n')      # raw 0x80: kills the ITERATION, not just json
        fh.write(_line("b").encode())
    c = _ScoreCache(str(p))
    assert "a" in c.mem and "b" in c.mem, "a bad byte must not abort the read"


def test_roundtrip_and_no_torn_writes(tmp_path):
    p = tmp_path / "c.jsonl"
    c = _ScoreCache(str(p))
    for i in range(50):
        c.put(f"k{i}", 0.5, "r", 10)
    reread = _ScoreCache(str(p))
    assert len(reread.mem) == 50 and reread.torn == 0
    assert reread.get("k7") == (0.5, "r", 10)


def test_concurrent_appends_are_not_torn(tmp_path):
    """Concurrent writers must emit WHOLE records -- no torn lines.

    Asserts only what the design promises. Record COUNT is deliberately not asserted:
    `put` is open/seek/write per call, so concurrent opens can land on the same offset and
    lose a record (measured: 8 writers x 40 -> 250 lines). That is a cache miss, i.e. a
    re-paid call, never a wrong score -- and the reader tolerates it. Tearing is the thing
    that used to be catastrophic, so tearing is what is pinned.""" 
    import threading
    p = tmp_path / "c.jsonl"
    caches = [_ScoreCache(str(p)) for _ in range(8)]      # 8 independent writers, one file
    def w(ci):
        for i in range(40):
            caches[ci].put(f"w{ci}-{i}", 0.5, "x" * 200, 10)
    ts = [threading.Thread(target=w, args=(i,)) for i in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    reread = _ScoreCache(str(p))
    assert reread.torn == 0, f"{reread.torn} torn records from concurrent appends"
    assert reread.mem, "some records must survive"
    for k, v in reread.mem.items():                  # every surviving record is intact
        assert v == (0.5, "x" * 200, 10), f"{k} came back mangled: {v!r}"
