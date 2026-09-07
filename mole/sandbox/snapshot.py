"""Sandbox snapshot/restore — capture the docker state of a seeded LabSandbox so it
can be restored for later runs without paying the seed cost again.

Why: the Phase 2a adversary grid runs hundreds of orchestrator episodes, each on a
clean copy of the seeded world. Without snapshotting, each cell re-runs `seed_company`
(minutes per cell × hundreds of cells = hours wasted). With snapshotting we seed once,
snapshot, then `restore_for_episode()` per cell.

What we capture:
  - Each container's filesystem changes via `docker commit` to a snapshot image.
  - Each volume's contents via `docker run --rm + tar` into a tarball.
  - Manifest JSON describing the snapshot (services, images, volumes, timestamp).

What we DON'T capture here (yet):
  - Per-manager in-process Python state for the non-containerized services
    (`model_registry`, `eval_server`, `secrets_store`, `org`, `plane`). They re-load
    deterministically from their YAMLs at process start so a snapshot of the docker
    side + a re-seed of the in-process side gives the same world. The per-manager
    state dump is a follow-up if we need to capture agent-mutated registry/secrets.

Usage:
  from mole.sandbox.snapshot import snapshot_sandbox, restore_sandbox
  await snapshot_sandbox(sandbox, "seeded_v1", out_dir="data/snapshots/seeded_v1")
  ...
  await restore_sandbox("data/snapshots/seeded_v1", new_session_id="adv-A0-s06-r1")
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import shlex
from pathlib import Path
from typing import Any

from .docker import _async_run

logger = logging.getLogger(__name__)

SNAPSHOT_IMAGE_PREFIX = "itb-snap"


def _safe(name: str) -> str:
    """Sanitize a name for use in docker tags / filenames."""
    return re.sub(r"[^a-zA-Z0-9._-]", "_", name).strip("._-")[:80] or "snap"


async def _docker_json(cmd: str) -> list[dict[str, Any]]:
    """Run a docker command that outputs JSON-per-line and parse it."""
    r = await _async_run(cmd, timeout_s=120)
    out: list[dict[str, Any]] = []
    for line in (r.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("docker_json: non-JSON line skipped: %s", line[:80])
    return out


async def _list_compose_containers(session_id: str) -> list[dict[str, Any]]:
    """All containers belonging to the compose project, running or stopped.

    Legacy fallback: filters by the `com.docker.compose.project` label. This is
    fragile — if compose normalized the project name (case/charset) the raw
    `session_id` won't match the label and this silently returns []. Prefer
    `_compose_container_ids` (enumerate via the compose project itself)."""
    cmd = (
        f"docker ps -a --filter label=com.docker.compose.project={shlex.quote(session_id)} "
        "--format '{{json .}}'"
    )
    return await _docker_json(cmd)


async def _compose_container_ids(
    session_id: str, compose_files: "list[Path]",
) -> list[str]:
    """Container IDs for a compose project, asked of compose itself.

    Robust where the label filter is not: `docker compose -p <id> -f ... ps`
    resolves the project exactly as `up` did, so it can't miss containers to a
    project-name normalization mismatch (the bug that left kimi's snapshots
    empty — the label filter returned zero, so snapshot_sandbox raised before
    writing anything)."""
    flags = " ".join(f"-f {shlex.quote(str(f))}" for f in compose_files)
    cmd = f"docker compose -p {shlex.quote(session_id)} {flags} ps -a -q"
    r = await _async_run(cmd, timeout_s=60)
    if r.return_code != 0:
        logger.warning("compose ps failed for project %s: %s",
                       session_id, (r.stderr or "").strip()[:200])
        return []
    return [ln.strip() for ln in (r.stdout or "").splitlines() if ln.strip()]


async def _inspect_container(cid: str) -> dict[str, Any]:
    r = await _async_run(f"docker inspect {shlex.quote(cid)}", timeout_s=60)
    data = json.loads(r.stdout)
    return data[0] if isinstance(data, list) and data else {}


async def snapshot_sandbox(
    session_id: str,
    name: str,
    *,
    out_dir: str | Path,
    compose_files: "list[Path] | None" = None,
) -> dict[str, Any]:
    """Snapshot every container in the sandbox's compose project to images + volume
    tarballs under `out_dir`. Returns the manifest dict (also written to out_dir).

    Pauses containers briefly during commit for a consistent point-in-time view; resumes
    them after. Safe to call against a live sandbox.

    `compose_files`: when given, containers are enumerated via the compose
    project itself (robust); otherwise the legacy label filter is used (fragile —
    see `_compose_container_ids`). Callers with a live sandbox should always pass
    `sandbox.compose_files`.
    """
    snap_name = _safe(name)
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    vols_dir = out / "volumes"
    vols_dir.mkdir(exist_ok=True)

    if compose_files:
        ids = await _compose_container_ids(session_id, compose_files)
    else:
        containers = await _list_compose_containers(session_id)
        ids = [c.get("ID") or c.get("Id") for c in containers
               if c.get("ID") or c.get("Id")]
    if not ids:
        # Loud + specific: an empty result here is exactly what silently left the
        # kimi snapshots empty. The caller records snapshot_ok=False so a
        # --restore-world resume won't trust this day.
        how = "compose-ps" if compose_files else "label-filter fallback (misses normalized project names)"
        raise RuntimeError(
            f"snapshot: no containers found for compose project {session_id!r} "
            f"via {how}. Nothing captured.")

    # pause so the filesystem + volumes are quiescent during commit/backup
    logger.info("snapshot %s: pausing %d container(s) for consistent capture",
                snap_name, len(ids))
    for cid in ids:
        await _async_run(f"docker pause {shlex.quote(cid)}", timeout_s=30, check=False)

    manifest: dict[str, Any] = {
        "name": snap_name,
        "source_session_id": session_id,
        "services": [],
    }
    try:
        seen_volumes: set[str] = set()
        for cid in ids:
            details = await _inspect_container(cid)
            # service name from the compose label
            labels = (details.get("Config") or {}).get("Labels") or {}
            service = labels.get("com.docker.compose.service") or details.get("Name", cid).lstrip("/")
            service_safe = _safe(service)
            image_tag = f"{SNAPSHOT_IMAGE_PREFIX}/{snap_name}/{service_safe}:latest"

            # commit container filesystem -> image
            logger.info("snapshot %s: commit %s -> %s", snap_name, service, image_tag)
            await _async_run(
                f"docker commit {shlex.quote(cid)} {shlex.quote(image_tag)}",
                timeout_s=600,
            )

            # capture volume mounts (anonymous + named). For each unique volume,
            # tar its contents to a host file via a tiny busybox helper.
            vol_records: list[dict[str, str]] = []
            for m in details.get("Mounts") or []:
                if m.get("Type") != "volume":
                    continue
                vname = m.get("Name") or ""
                if not vname or vname in seen_volumes:
                    continue
                seen_volumes.add(vname)
                mountpoint = m.get("Destination") or "/"
                tar_name = f"{_safe(vname)}.tar.gz"
                tar_path = vols_dir / tar_name
                logger.info("snapshot %s: backup volume %s (%s)", snap_name, vname, mountpoint)
                # busybox is tiny; mount the source volume read-only + the out dir rw.
                helper = (
                    f"docker run --rm "
                    f"-v {shlex.quote(vname)}:/source:ro "
                    f"-v {shlex.quote(str(vols_dir))}:/backup "
                    f"busybox sh -c "
                    + shlex.quote(f"cd /source && tar czf /backup/{tar_name} .")
                )
                await _async_run(helper, timeout_s=1800)
                vol_records.append({
                    "volume_name": vname,
                    "tarball": tar_name,
                    "mount_point": mountpoint,
                })

            manifest["services"].append({
                "service": service,
                "image": image_tag,
                "container_image_original": details.get("Config", {}).get("Image"),
                "volumes": vol_records,
            })
    finally:
        for cid in ids:
            await _async_run(f"docker unpause {shlex.quote(cid)}", timeout_s=30, check=False)

    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("snapshot %s: wrote %s (services=%d, volumes=%d)",
                snap_name, manifest_path, len(manifest["services"]), len(seen_volumes))
    return manifest


async def restore_volumes_in_place(snapshot_dir: str | Path) -> dict[str, Any]:
    """Restore a snapshot's volumes back under their ORIGINAL names, so bringing
    the SAME compose project (same session_id) back up reattaches to the restored
    world — no volume-name remapping or compose override needed.

    This is the resume path (vs `restore_sandbox`, which clones into a fresh
    project for parallel sweep cells). Pre-req: the original containers are gone
    (a resume after a crash) so the volumes can be recreated cleanly; we `rm -f`
    each target volume first so the tar extract lands on an empty volume.

    Returns {restored: [volume names], snapshot_dir, source_session_id}.

    NOTE: validated against real docker before the production resume
    relies on it — the tar round-trip + GitLab postgres/repo volumes need a live
    smoke (a unit test can't cover container bring-up).
    """
    snap = Path(snapshot_dir).resolve()
    manifest = json.loads((snap / "manifest.json").read_text(encoding="utf-8"))
    vols_dir = snap / "volumes"
    restored: list[str] = []
    for svc in manifest.get("services", []):
        for v in svc.get("volumes", []):
            name = v["volume_name"]
            if name in restored:
                continue
            tarball = vols_dir / v["tarball"]
            if not tarball.exists():
                raise FileNotFoundError(f"restore_in_place: tarball missing: {tarball}")
            # Drop any stale volume, recreate empty, extract into it. `rm` fails
            # harmlessly if the volume is absent or still referenced (we expect
            # it gone post-crash); surface but don't abort on that.
            await _async_run(f"docker volume rm {shlex.quote(name)}",
                             timeout_s=30, check=False)
            await _async_run(f"docker volume create {shlex.quote(name)}", timeout_s=30)
            helper = (
                f"docker run --rm "
                f"-v {shlex.quote(name)}:/dest "
                f"-v {shlex.quote(str(vols_dir))}:/backup:ro "
                f"busybox sh -c "
                + shlex.quote(f"cd /dest && tar xzf /backup/{v['tarball']}")
            )
            await _async_run(helper, timeout_s=1800)
            logger.info("restore_in_place: restored volume %s from %s",
                        name, tarball.name)
            restored.append(name)
    return {
        "restored": restored,
        "snapshot_dir": str(snap),
        "source_session_id": manifest.get("source_session_id"),
    }


async def restore_sandbox(
    snapshot_dir: str | Path,
    *,
    new_session_id: str,
) -> dict[str, Any]:
    """Restore a snapshot's volumes under fresh project-prefixed names. Returns a
    mapping {original_volume: new_volume} suitable for wiring into a compose override.

    Note: this restores the VOLUME DATA into new volumes named
    `<new_session_id>_<original_volume_safe>`. Bringing the containers up against
    those restored volumes is the next step — typically done by a compose override
    that maps each service's volume to the restored one. That wiring lives in the
    caller (e.g. an adversary-grid runner that generates the override per-cell).
    """
    snap = Path(snapshot_dir).resolve()
    manifest = json.loads((snap / "manifest.json").read_text(encoding="utf-8"))
    vols_dir = snap / "volumes"
    mapping: dict[str, str] = {}

    for svc in manifest.get("services", []):
        for v in svc.get("volumes", []):
            orig = v["volume_name"]
            if orig in mapping:
                continue
            tarball = vols_dir / v["tarball"]
            if not tarball.exists():
                raise FileNotFoundError(f"restore: tarball missing: {tarball}")
            new_name = _safe(f"{new_session_id}_{orig}")
            logger.info("restore: create volume %s and restore %s", new_name, tarball.name)
            await _async_run(f"docker volume create {shlex.quote(new_name)}", timeout_s=30)
            helper = (
                f"docker run --rm "
                f"-v {shlex.quote(new_name)}:/dest "
                f"-v {shlex.quote(str(vols_dir))}:/backup:ro "
                f"busybox sh -c "
                + shlex.quote(f"cd /dest && tar xzf /backup/{v['tarball']}")
            )
            await _async_run(helper, timeout_s=1800)
            mapping[orig] = new_name

    return {
        "manifest": manifest,
        "volume_remap": mapping,
        "new_session_id": new_session_id,
    }
