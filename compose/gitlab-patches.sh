#!/bin/bash
# Pre-launch patches for the TAC GitLab Omnibus image.
#
# Bind-mounted into the container by compose/lab.yaml and invoked as
# the container's command before exec'ing /assets/wrapper. Keeping
# this as a real script (vs. inlining sed in the compose YAML) avoids
# four nested layers of quoting hell (YAML → shell list arg → bash -c
# string → sed s| | |).
#
# We apply two independent patches:
#
# 1. runsvdir-start sysctl write
#    Line 37 writes "1000000" > /proc/sys/fs/file-max, which fails on
#    non-privileged containers (/proc/sys is read-only). The failure
#    cascades: runsv can't allocate sockets → never registers
#    workhorse → chef's `sv restart gitlab-workhorse` fails →
#    container exits(1). Patch replaces line 37 with `true`.
#
# 2. /assets/wrapper race between runsvdir-start and chef reconfigure
#    The wrapper does:
#        GITLAB_OMNIBUS_CONFIG= /opt/gitlab/embedded/bin/runsvdir-start &
#        gitlab-ctl reconfigure
#    i.e., runsvdir-start is launched in the background and chef is
#    invoked immediately. On a loaded host (or just unlucky timing),
#    chef reaches `sv restart gitlab-workhorse` before runsv has
#    registered services. Patch inserts an `until pgrep -x runsv`
#    wait + a 5s settle BEFORE the gitlab-ctl reconfigure line.
#
# This script is idempotent — sed -i is a no-op if the pattern's
# already replaced. So restarting the container is safe.

set -euo pipefail

echo "[gitlab-patches] applying runsvdir-start sysctl-write patch"
sed -i 's|echo "1000000" > /proc/sys/fs/file-max|true|' \
    /opt/gitlab/embedded/bin/runsvdir-start

echo "[gitlab-patches] applying wrapper runsv-wait + chef-retry patch"
# Two layers of robustness in the wrapper patch:
#  (1) wait for ANY runsv to appear before chef starts
#  (2) retry chef up to 3 times on failure with 15s settle between
#      tries — chef's runit_service[gitlab-workhorse] races runsvdir's
#      5s service-polling interval (chef creates the workhorse symlink
#      and immediately `sv restart`s it; runsvdir hasn't polled yet so
#      no runsv for workhorse exists, command fails). A 15s settle
#      lets runsvdir poll twice and pick up the new service.
#  (3) if 3 attempts still fail, log + continue — runsvdir keeps
#      running and the rest of the wrapper (gitlab-ctl tail, wait)
#      stays alive so we can inspect.
sed -i 's|^gitlab-ctl reconfigure$|echo "[wrapper] waiting for runsv to register services..."; until pgrep -x runsv > /dev/null 2>\&1; do sleep 1; done; echo "[wrapper] runsv present; settling 5s"; sleep 5; attempts=0; until gitlab-ctl reconfigure; do attempts=$((attempts+1)); if [ $attempts -ge 3 ]; then echo "[wrapper] reconfigure failed 3x; continuing anyway"; break; fi; echo "[wrapper] reconfigure attempt $attempts failed; sleeping 15s and retrying"; sleep 15; done|' \
    /assets/wrapper

echo "[gitlab-patches] verifying patches:"
sed -n '37p' /opt/gitlab/embedded/bin/runsvdir-start
grep -n 'until pgrep -x runsv' /assets/wrapper | head -2

echo "[gitlab-patches] handing off to /assets/wrapper"
exec /assets/wrapper
