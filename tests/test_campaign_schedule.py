"""Deterministic tests for the multi-day campaign scheduler (no network)."""
from __future__ import annotations

import random

from mole.generator.campaign import PHASE_DECOMP, build_campaign, build_single_day

DATES = [f"2026-04-{d:02d}" for d in range(6, 6 + 20)]   # 20 working-day-ish strings


def _idx(dates):
    return {d: i for i, d in enumerate(dates)}


def test_stage_before_trigger_and_in_window():
    ph = build_campaign("10_safety_sabotage", DATES, random.Random(0))
    assert ph, "expected a non-empty campaign for a decomposable threat"
    idx = _idx(DATES)
    assert all(p["date"] in idx for p in ph)                 # every phase date in window
    stages = [idx[p["date"]] for p in ph if p["phase"] == "stage"]
    triggers = [idx[p["date"]] for p in ph if p["phase"] == "trigger"]
    assert len(stages) == len(triggers) >= 1
    # each stage has a trigger strictly after it
    for s in stages:
        assert any(t > s for t in triggers)


def test_gap_within_range():
    dates = [f"2026-05-{d:02d}" for d in range(1, 29)]       # 28 days, plenty of room
    ph = build_campaign("06_weight_exfil", dates, random.Random(3))
    idx = _idx(dates)
    stages = sorted(idx[p["date"]] for p in ph if p["phase"] == "stage")
    triggers = sorted(idx[p["date"]] for p in ph if p["phase"] == "trigger")
    for s in stages:
        assert any(2 <= (t - s) <= 6 for t in triggers), f"no trigger 2..6 days after stage {s}"


def test_reproducible_for_same_seed_and_varies_across_seeds():
    a = build_campaign("01_ip_theft", DATES, random.Random(42))
    b = build_campaign("01_ip_theft", DATES, random.Random(42))
    assert a == b                                            # deterministic
    c = build_campaign("01_ip_theft", DATES, random.Random(7))
    assert a != c                                            # different seed -> different schedule


def test_non_decomposable_threats_are_single_day():
    # Non-decomposable threats now yield ONE single agent-day (K=1), not [] —
    # so multi-day mode never falls back to the per-session coin-flip.
    for scn in ("c1_release_bypass", "c2_split_exfil", "totally_unknown"):
        ph = build_campaign(scn, DATES, random.Random(0))
        assert len(ph) == 1, scn
        assert ph[0]["phase"] == "single" and ph[0]["directive"] is None, scn


def test_04_and_05_now_decompose():
    # 04/05 were single-act; they now have stage->trigger decompositions.
    for scn in ("04_fraud", "05_sabotage"):
        assert [p["phase"] for p in build_campaign(scn, DATES, random.Random(0))] == ["stage", "trigger"]


def test_short_window_degrades_gracefully():
    # A window too short for a stage->trigger gap -> one single agent-day (K=1).
    one = build_campaign("01_ip_theft", DATES[:1], random.Random(0))
    assert len(one) == 1 and one[0]["phase"] == "single"
    # An empty window -> empty schedule.
    assert build_campaign("01_ip_theft", [], random.Random(0)) == []


def test_directives_match_phase_decomp():
    ph = build_campaign("08_data_poisoning", DATES, random.Random(1))
    assert ph
    for p in ph:
        assert p["directive"] == PHASE_DECOMP["08_data_poisoning"][p["phase"]]
        assert p["directive"]                                # non-empty


def test_all_decomposable_threats_have_both_directives():
    for scn, d in PHASE_DECOMP.items():
        assert d.get("stage") and d.get("trigger"), scn


# --- deterministic single-day mode (build_single_day) ---

def test_single_day_k_distinct_self_contained_days():
    days = build_single_day(DATES, random.Random(0), k=4)
    assert len(days) == 4
    assert len({d["date"] for d in days}) == 4                 # distinct days
    assert all(d["date"] in DATES for d in days)               # in window
    assert all(d["phase"] == "single" and d["directive"] for d in days)   # completion directive


def test_single_day_reproducible_and_varies_by_seed():
    a = build_single_day(DATES, random.Random(42), k=3)
    b = build_single_day(DATES, random.Random(42), k=3)
    assert a == b                                              # deterministic per seed
    c = build_single_day(DATES, random.Random(7), k=3)
    assert a != c                                              # per-insider jitter -> different days


def test_single_day_k_capped_to_window_and_empty():
    capped = build_single_day(DATES[:2], random.Random(0), k=5)
    assert len(capped) == 2                                    # k capped to window length
    assert build_single_day([], random.Random(0), k=4) == []  # empty window -> []
