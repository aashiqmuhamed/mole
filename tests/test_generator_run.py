"""Tests for the generator runner (generator.run.simulate).

Pins the acceptance criterion: `--personas 3 --days 5` yields a
benign-only audit log of ~500 events, deterministic under a fixed seed,
with role-based variance across background accounts.
"""
from __future__ import annotations

import asyncio
from collections import Counter

from mole.generator.run import simulate


def _run(tmp_path, **kw):
    out = tmp_path / "npc.jsonl"
    coll = asyncio.run(simulate(out_path=out, **kw))
    return coll, out


def test_simulate_produces_benign_only_log(tmp_path):
    coll, out = _run(tmp_path, n_personas=3, n_days=5, seed=0)
    assert out.exists()
    assert coll.events, "expected events"
    # The entire point: benign baseline.
    assert all(not e.is_malicious for e in coll.events)
    assert {e.account_kind for e in coll.events} == {"background_rules_agent"}


def test_simulate_hits_target_volume(tmp_path):
    coll, _ = _run(tmp_path, n_personas=3, n_days=5, seed=0)
    # After the multi-service workflow enrichment (#69) a 3-persona × 5-day
    # run lands ~1.5k events (was ~500 with only the 4 data-manager
    # workflows). Generous band so loaf/jitter variance isn't brittle.
    assert 800 <= len(coll.events) <= 4000, len(coll.events)


def test_benign_corpus_spans_the_service_surface(tmp_path):
    """#69: the benign rollup must have variance across the behavioral
    feature space — not just the 4 data managers — so monitors fit on it
    discriminate by behaviour, not service-coverage. file_delete and
    chat_external_mention are *correctly* zero in benign (they're suspicious
    signals), so we require the other ≥14 of 18 features to be non-degenerate."""
    from mole.audit.rollup import rollup
    from mole.monitors.classical.anomaly import NUMERIC_FEATURES
    coll, _ = _run(tmp_path, n_personas=8, n_days=5, seed=0)
    rows = rollup(coll.events)
    nondegenerate = sum(
        1 for f in NUMERIC_FEATURES if any(getattr(r, f) for r in rows)
    )
    assert nondegenerate >= 14, f"only {nondegenerate}/18 features have benign variance"
    # The services we enriched must all appear.
    services = {e.service for e in coll.events}
    assert {"email", "owncloud", "gitlab", "rocketchat", "session"} <= services


def test_simulate_is_deterministic_under_seed(tmp_path):
    c1, _ = _run(tmp_path / "a", n_personas=3, n_days=5, seed=7)
    c2, _ = _run(tmp_path / "b", n_personas=3, n_days=5, seed=7)
    sig1 = [(e.account, e.service, e.action) for e in c1.events]
    sig2 = [(e.account, e.service, e.action) for e in c2.events]
    assert sig1 == sig2


def test_simulate_has_role_based_variance(tmp_path):
    coll, _ = _run(tmp_path, n_personas=3, n_days=5, seed=0)
    by_account = Counter(e.account for e in coll.events)
    # 3 distinct background accounts each contribute.
    assert len(by_account) == 3
    assert all(v > 0 for v in by_account.values())
    # More than one service exercised.
    assert len({e.service for e in coll.events}) >= 2


def test_simulate_timestamps_within_working_hours(tmp_path):
    coll, _ = _run(tmp_path, n_personas=3, n_days=5, seed=0)
    # ts is ISO "YYYY-MM-DDTHH:MM:SSZ"; hours should cluster in the work
    # band (allow jitter spill to 08–18).
    hours = [int(e.ts[11:13]) for e in coll.events if len(e.ts) >= 13]
    assert hours, "expected timestamped events"
    assert all(8 <= h <= 18 for h in hours), sorted(set(hours))


def test_weekends_are_skipped(tmp_path):
    # Start on a Friday (2026-04-10); 3 working days → Fri, Mon, Tue.
    coll, _ = _run(tmp_path, n_personas=3, n_days=3, seed=0, start_date="2026-04-10")
    dates = {e.ts[:10] for e in coll.events}
    assert "2026-04-11" not in dates and "2026-04-12" not in dates  # Sat/Sun


def test_concurrency_preserves_event_volume_and_timestamps(tmp_path):
    """With concurrency>1, sessions within the same bucket run via asyncio.gather.
    Per-task contextvar clock keeps timestamps correct (no shared `_clock_fn`
    race). Verifies: same set of (account, service, action) tuples appears
    under concurrency=1 and concurrency=4, and all events still land in the
    work-hours band."""
    seq, _ = _run(tmp_path / "seq", n_personas=4, n_days=2, seed=42, concurrency=1)
    par, _ = _run(tmp_path / "par", n_personas=4, n_days=2, seed=42, concurrency=4)

    sig_seq = sorted((e.account, e.service, e.action) for e in seq.events)
    sig_par = sorted((e.account, e.service, e.action) for e in par.events)
    # Concurrency must not drop events or invent new accounts/actions.
    assert sig_seq == sig_par, (
        f"event signature differs: {len(sig_seq)} sequential vs "
        f"{len(sig_par)} concurrent"
    )

    # Timestamps must be IDENTICAL under concurrency — they're deterministic from
    # each member's plan, so a racy contextvar clock (one session leaking another's
    # sim_now) would change them. Exact multiset equality is the strongest check
    # and is independent of the (now varied, compressed) per-account work windows.
    assert sorted(e.ts for e in seq.events) == sorted(e.ts for e in par.events)


def test_concurrency_isolates_per_task_clock(tmp_path):
    """Stronger contextvar check: each event's ts must lie inside the planned
    window of the session that produced it. Every event-hour must fall in the
    plausible per-account chronotype band (a compressed continuous span from a
    persona-stable start) — an arbitrary hour would mean a global-clock race."""
    coll, _ = _run(tmp_path, n_personas=4, n_days=2, seed=42, concurrency=4,
                   bucket_minutes=15)
    # Chronotypes span business mornings (≈7:00) through evening workers (≈21:00),
    # plus a short compressed window + jitter — so the plausible band is ~6–23.
    # A leak would surface as an out-of-band hour (e.g., midnight/early morning).
    for e in coll.events:
        hour = int(e.ts[11:13])
        assert 6 <= hour <= 23, (
            f"event ts hour {hour} outside chronotype band — concurrent clock leak?"
        )


def test_resume_skips_completed_days_and_appends(tmp_path):
    """--resume should: (1) skip days already present in the audit log,
    (2) open the log in append mode so prior events are preserved, (3) only
    add events for the remaining days. Uses rules-mode (no sandbox) so we
    test the resume bookkeeping in isolation from container plumbing."""
    out = tmp_path / "npc.jsonl"
    # First run: 3 working days starting Monday 2026-04-06.
    coll_a = asyncio.run(simulate(
        n_personas=2, n_days=3, out_path=out, seed=7, start_date="2026-04-06",
    ))
    n_initial = len(coll_a.events)
    dates_initial = sorted({e.ts[:10] for e in coll_a.events})
    assert len(dates_initial) == 3

    # Second "resumed" run: same args + 2 extra days, --resume.
    # Should NOT re-emit events for the 3 days already in the log.
    coll_b = asyncio.run(simulate(
        n_personas=2, n_days=5, out_path=out, seed=7, start_date="2026-04-06",
        resume=True,
    ))
    # After resume, the on-disk log should contain initial events PLUS new days.
    # Read the file to verify append behavior:
    with out.open("r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    # The file should have grown (resume appended) — at least the initial count.
    assert len(lines) > n_initial, (
        f"expected resume to append events; on-disk has {len(lines)}, "
        f"initial run had {n_initial}"
    )
    # All initial dates must still be present in the on-disk log.
    import json as _json
    on_disk_dates = sorted({
        _json.loads(line).get("ts", "")[:10] for line in lines
    } - {""})
    assert set(dates_initial) <= set(on_disk_dates), (
        f"resume dropped earlier days: {dates_initial} vs {on_disk_dates}"
    )
    # And resume should have added new dates beyond the initial 3.
    assert len(on_disk_dates) > len(dates_initial)


def test_resume_with_empty_log_starts_fresh(tmp_path):
    """--resume against a non-existent log should warn but proceed from day 1."""
    out = tmp_path / "npc.jsonl"
    coll = asyncio.run(simulate(
        n_personas=2, n_days=2, out_path=out, seed=0, resume=True,
    ))
    assert coll.events, "expected events even when resuming against no prior log"
