"""Tests for the multi-day campaign scheduler (generator/campaign.py).

Focus: COLLUSION pair sequencing. c1/c2 are non-decomposable, so each insider used to
draw an independent random attack-day -> ~half the time the co-signer/exfiltrator landed
*before* the requester/puller and the cross-day handoff was impossible. build_campaign now
places collusion roles by rank (requester -> early half, co-signer -> late half), so the
first role is always strictly before the second regardless of per-insider seeds.
"""
import random

from mole.generator.campaign import build_campaign, build_benign_arc


def _day_index(schedule, sim_dates):
    assert len(schedule) == 1, schedule
    return sim_dates.index(schedule[0]["date"])


def test_collusion_pair_sequenced_requester_before_cosigner():
    # 4-day window, the multi-day default. The two partners use DIFFERENT seeds
    # (they're different personas, rng = Random(seed + persona_index)); the role
    # window must still guarantee requester-before-cosigner every time.
    sim_dates = [f"2026-04-{d:02d}" for d in (6, 7, 8, 9)]
    for seed in range(200):
        for scen, r0, r1 in (("c1_release_bypass", "approver_1", "approver_2"),
                             ("c2_split_exfil", "read_access", "egress_access")):
            d_req = _day_index(build_campaign(scen, sim_dates, random.Random(seed), role=r0), sim_dates)
            d_cos = _day_index(build_campaign(scen, sim_dates, random.Random(seed + 1), role=r1), sim_dates)
            assert d_req < d_cos, f"{scen}: requester day {d_req} !< co-signer day {d_cos} (seed {seed})"


def test_collusion_sequencing_holds_for_small_windows():
    for n in (2, 3, 5, 7):
        sim_dates = [f"2026-04-{6 + i:02d}" for i in range(n)]
        for seed in range(60):
            d_req = _day_index(build_campaign("c1_release_bypass", sim_dates,
                                              random.Random(seed), role="approver_1"), sim_dates)
            d_cos = _day_index(build_campaign("c1_release_bypass", sim_dates,
                                              random.Random(seed + 7), role="approver_2"), sim_dates)
            assert d_req < d_cos, f"n={n} seed={seed}: requester {d_req} !< co-signer {d_cos}"


def test_no_role_keeps_legacy_random_single_day():
    # role=None (solo non-decomposable / benign) -> unchanged single random day.
    sim_dates = [f"2026-04-{6 + i:02d}" for i in range(4)]
    sched = build_campaign("c1_release_bypass", sim_dates, random.Random(0), role=None)
    assert len(sched) == 1 and sched[0]["phase"] == "single" and sched[0]["directive"] is None


def test_solo_decomposable_still_stage_then_trigger():
    # A decomposable solo threat is unaffected by the collusion change: stage<trigger.
    sim_dates = [f"2026-04-{6 + i:02d}" for i in range(8)]
    sched = build_campaign("01_ip_theft", sim_dates, random.Random(1))
    phases = {p["phase"] for p in sched}
    assert {"stage", "trigger"} <= phases
    s = sim_dates.index(next(p["date"] for p in sched if p["phase"] == "stage"))
    t = sim_dates.index(next(p["date"] for p in sched if p["phase"] == "trigger"))
    assert s < t


def test_collusion_roles_get_act_phase_and_directive():
    # Both collusion roles must run as tagged attack sessions ("act") WITH a directive --
    # without it the personas alone defaulted to benign cover (c2).
    sim_dates = [f"2026-04-{6 + i:02d}" for i in range(4)]
    for scen, r0, r1 in (("c1_release_bypass", "approver_1", "approver_2"),
                         ("c2_split_exfil", "read_access", "egress_access")):
        for role in (r0, r1):
            p = build_campaign(scen, sim_dates, random.Random(0), role=role)[0]
            assert p["phase"] == "act", p
            assert p["directive"], f"{scen}/{role}: no directive"


def test_c2_roles_name_same_handoff_location():
    # No cross-agent recap -> both c2 roles must point at the SAME shared folder, else the
    # exfiltrator reads an empty location.
    sim_dates = [f"2026-04-{6 + i:02d}" for i in range(4)]
    loc = "/shared/model-cards/v17-stage/"
    d_stage = build_campaign("c2_split_exfil", sim_dates, random.Random(0), role="read_access")[0]["directive"]
    d_exfil = build_campaign("c2_split_exfil", sim_dates, random.Random(0), role="egress_access")[0]["directive"]
    assert loc in d_stage and loc in d_exfil


# --- benign multi-day arcs (default-off feature) ---

def test_benign_arc_strictly_increasing_days():
    sim_dates = [f"2026-04-{6 + i:02d}" for i in range(7)]
    arc = {"name": "x", "steps": ["s0", "s1", "s2"]}
    for seed in range(200):
        sched = build_benign_arc(arc, sim_dates, random.Random(seed))
        assert len(sched) == 3
        days = [sim_dates.index(p["date"]) for p in sched]
        assert days == sorted(days) and len(set(days)) == 3, (seed, days)
        assert [p["step_idx"] for p in sched] == [0, 1, 2]
        assert [p["directive"] for p in sched] == ["s0", "s1", "s2"]


def test_benign_arc_empty_and_short_window():
    sim_dates = [f"2026-04-{6 + i:02d}" for i in range(7)]
    assert build_benign_arc({"steps": []}, sim_dates, random.Random(0)) == []
    assert build_benign_arc({"name": "x", "steps": ["a"]}, [], random.Random(0)) == []
    # window shorter than the step count -> truncate to n, still strictly ordered
    sched = build_benign_arc({"name": "x", "steps": ["a", "b", "c", "d", "e"]},
                             [f"2026-04-{6 + i:02d}" for i in range(2)], random.Random(1))
    assert len(sched) == 2 and sched[0]["date"] < sched[1]["date"]


def test_benign_arc_reproducible_per_seed():
    sim_dates = [f"2026-04-{6 + i:02d}" for i in range(10)]
    arc = {"name": "x", "steps": ["a", "b", "c"]}
    assert build_benign_arc(arc, sim_dates, random.Random(42)) == build_benign_arc(arc, sim_dates, random.Random(42))


def test_benign_arc_contiguous_not_spread_on_long_window():
    # On a 30-day window a 3-step arc must CLUSTER (<= 2 gaps * max 3 = 6 days), not span the month.
    sim_dates = [f"2026-04-{1 + i:02d}" for i in range(30)]
    arc = {"name": "x", "steps": ["a", "b", "c"]}
    for seed in range(100):
        days = [sim_dates.index(p["date"]) for p in build_benign_arc(arc, sim_dates, random.Random(seed))]
        assert days[-1] - days[0] <= 6, (seed, days)
