"""EmailManager — SMTP + IMAP backend.

Talks to a GreenMail-flavoured mail server (SMTP on 3025, IMAP on 3143,
auth disabled) running in the per-session sandbox. Reached from the host
on dynamic loopback ports — sandbox.ports[3025] / sandbox.ports[3143].

Method surface (covers threats 01 IP-theft, 02 secret-exfil, 06 weight-
exfil, 07 eval-manip — all need mail as either a delivery channel for the
cover task or as an external-egress path):

  send_email(from_user, to, subject, body, *, cc=None, attachments=None) → message_id
  read_inbox(user, *, max_count=10, folder="INBOX") → [{id, from, to, subject, body, ts}]
  find_emails(user, *, subject_contains=None, from_contains=None) → [<as above>]
  clear_folder(user, *, folder="INBOX")
  forward_email(user, message_id, to)
"""
from __future__ import annotations

import asyncio
import email
import imaplib
import logging
import os
import smtplib
from email.message import EmailMessage
from typing import Any, Iterable

from ..base import StateManager

logger = logging.getLogger(__name__)


@StateManager.register("email")
class EmailManager(StateManager):
    DEFAULT_SMTP_PORT = 3025
    DEFAULT_IMAP_PORT = 3143
    DEFAULT_HOST = "127.0.0.1"
    DEFAULT_DOMAIN = "agentlab.local"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._smtp_host: str = ""
        self._smtp_port: int = 0
        self._imap_host: str = ""
        self._imap_port: int = 0
        self._domain: str = ""
        # SMTP / IMAP factories are kept as attributes so unit tests can
        # swap them for fakes without touching the network.
        self._smtp_factory = smtplib.SMTP
        self._imap_factory = imaplib.IMAP4

    async def setup(self, *, sandbox) -> None:
        host = self.config.get("host") or os.environ.get("EMAIL_HOST") or self.DEFAULT_HOST
        smtp_port = (
            self.config.get("smtp_port")
            or int(os.environ.get("EMAIL_SMTP_PORT", "0") or 0)
            or (sandbox.ports or {}).get(self.DEFAULT_SMTP_PORT)
        )
        imap_port = (
            self.config.get("imap_port")
            or int(os.environ.get("EMAIL_IMAP_PORT", "0") or 0)
            or (sandbox.ports or {}).get(self.DEFAULT_IMAP_PORT)
        )
        if smtp_port is None or imap_port is None:
            raise RuntimeError(
                f"email backend needs both SMTP ({self.DEFAULT_SMTP_PORT}) and "
                f"IMAP ({self.DEFAULT_IMAP_PORT}) ports in sandbox.ports or env"
            )
        self._smtp_host, self._smtp_port = host, int(smtp_port)
        self._imap_host, self._imap_port = host, int(imap_port)
        self._domain = (
            self.config.get("domain")
            or os.environ.get("EMAIL_DOMAIN")
            or self.DEFAULT_DOMAIN
        )
        logger.info(
            "email manager up: smtp=%s:%d imap=%s:%d domain=%s",
            self._smtp_host, self._smtp_port, self._imap_host, self._imap_port,
            self._domain,
        )

    def _addr_for(self, user: str) -> str:
        """Normalise `user` to a full email address.

        IMAP servers (GreenMail in particular) key mailboxes by the
        full address. Tasks naturally pass account ids like
        "bob.li"; this helper appends `@<domain>` so IMAP login lands
        on the right mailbox.
        """
        return user if "@" in user else f"{user}@{self._domain}"

    async def cleanup(self) -> None:
        # No long-lived connections — each call opens its own SMTP/IMAP session.
        pass

    async def reset(self, *, sandbox) -> None:
        """Wipe every mailbox the agent could have touched.

        Strategy: walk the LIST namespace, login as each existing user,
        select INBOX, flag-deleted everything, expunge. Idempotent and
        survives GreenMail's auth-disabled mode where any creds work.

        State that survives reset: GreenMail mailbox metadata itself
        (the mailboxes still exist, just empty). That's fine — next
        send to a known account repopulates. We don't tear down
        the container.
        """
        addresses = self._known_addresses()
        for addr in addresses:
            try:
                await self.clear_folder(user=addr, folder="INBOX")
            except Exception as exc:                       # noqa: BLE001
                logger.warning("email.reset: clear %s failed: %s", addr, exc)
        # Connections are short-lived; nothing else to do. Re-run setup
        # for symmetry with other backends, in case ports rotated.
        await self.setup(sandbox=sandbox)

    def _known_addresses(self) -> list[str]:
        """Mailbox addresses we should consider cleaning.

        Pulls from config['known_addresses'] when set (the pool can pass
        them in from the loaded org template). Falls back to a hardcoded
        set of seeded accounts + the admin sender used by seed_org.
        """
        from_config = self.config.get("known_addresses")
        if from_config:
            return list(from_config)
        # Conservative default: the org_template.yaml seed list + admin.
        return [
            "admin@agentlab.local",
            "alice.kim@agentlab.local",
            "bob.li@agentlab.local",
            "dan.r@agentlab.local",
            "frank.s@agentlab.local",
            "grace.t@agentlab.local",
            "hannah.m@agentlab.local",
            "ivan.o@agentlab.local",
            "julian.x@agentlab.local",
            "kara.p@agentlab.local",
        ]

    # ── send ────────────────────────────────────────────────────────

    async def send_email(
        self,
        *,
        from_user: str | None = None,
        sender: str | None = None,
        from_: str | None = None,
        to: str | list[str] | None = None,
        recipients: str | list[str] | None = None,
        subject: str = "",
        body: str | None = None,
        content: str | None = None,
        text: str | None = None,
        cc: str | list[str] | None = None,
        attachments: list[dict[str, Any]] | None = None,
        in_reply_to: str | None = None,
        **_extra: Any,
    ) -> dict[str, Any]:
        """Send one email. Pass `to` (recipient address or list), `subject`, and
        `body`; optional `cc`, `attachments`, and `in_reply_to` (a prior Message-ID
        to thread this as a reply). Returns {message_id, to, bytes}."""
        from_user = from_user or sender or from_ or "system@agentlab.local"
        to = to or recipients or []
        body = body or content or text or ""
        msg = self._build_message(
            from_user=from_user, to=to, subject=subject, body=body,
            cc=cc, attachments=attachments, in_reply_to=in_reply_to,
        )

        def _call() -> dict[str, Any]:
            with self._smtp_factory(self._smtp_host, self._smtp_port) as s:
                s.send_message(msg)
            return {
                "message_id": msg["Message-ID"] or "",
                "to": _as_list(to),
                "cc": _as_list(cc) if cc else [],
                "bytes": len(msg.as_bytes()),
            }
        return await asyncio.to_thread(_call)

    # ── receive ────────────────────────────────────────────────────

    async def read_inbox(
        self, *, user: str | None = None,
        max_count: int = 10, folder: str = "INBOX",
        **_extra: Any,
    ) -> list[dict[str, Any]]:
        """Fetch up to `max_count` messages from user's inbox, newest first."""
        if user is None:
            return []
        def _call() -> list[dict[str, Any]]:
            with self._imap_factory(self._imap_host, self._imap_port) as m:
                addr = self._addr_for(user)
                m.login(addr, addr)             # GreenMail accepts any creds; addr=full email
                typ, _sel = m.select(folder)
                if typ != "OK":
                    # Agent asked for a folder GreenMail doesn't have (e.g. "Sent",
                    # "Archive"); fall back to INBOX so SEARCH isn't issued in AUTH
                    # state (was: "command SEARCH illegal in state AUTH").
                    typ, _sel = m.select("INBOX")
                    if typ != "OK":
                        return []
                typ, ids = m.search(None, "ALL")
                if typ != "OK":
                    return []
                id_list = (ids[0] or b"").split()
                if not id_list:
                    return []
                # Newest first; cap at max_count.
                id_list = list(reversed(id_list))[:max_count]
                out: list[dict[str, Any]] = []
                for mid in id_list:
                    typ, data = m.fetch(mid, "(RFC822)")
                    if typ != "OK" or not data or not data[0]:
                        continue
                    raw = data[0][1] if isinstance(data[0], tuple) else data[0]
                    out.append(_summarise_message(raw, message_id=mid.decode("ascii")))
                return out
        return await asyncio.to_thread(_call)

    async def find_emails(
        self,
        *,
        user: str | None = None,
        subject_contains: str | None = None,
        from_contains: str | None = None,
        max_count: int = 100,
        folder: str = "INBOX",
        **_extra: Any,
    ) -> list[dict[str, Any]]:
        """Search a user's inbox by substring. Pass `user`, and optionally
        `subject_contains` and/or `from_contains` to filter."""
        if user is None:
            return []
        msgs = await self.read_inbox(user=user, max_count=max_count, folder=folder)
        def _match(m: dict[str, Any]) -> bool:
            if subject_contains and subject_contains.lower() not in m["subject"].lower():
                return False
            if from_contains and from_contains.lower() not in m["from"].lower():
                return False
            return True
        return [m for m in msgs if _match(m)]

    async def clear_folder(self, *, user: str, folder: str = "INBOX") -> int:
        """Delete every message in `folder`. Returns the count removed."""
        def _call() -> int:
            with self._imap_factory(self._imap_host, self._imap_port) as m:
                addr = self._addr_for(user)
                m.login(addr, addr)
                typ, _sel = m.select(folder)
                if typ != "OK":
                    # Agent asked for a folder GreenMail doesn't have (e.g. "Sent",
                    # "Archive"); fall back to INBOX so SEARCH isn't issued in AUTH
                    # state (was: "command SEARCH illegal in state AUTH").
                    typ, _sel = m.select("INBOX")
                    if typ != "OK":
                        return []
                typ, ids = m.search(None, "ALL")
                if typ != "OK":
                    return 0
                id_list = (ids[0] or b"").split()
                for mid in id_list:
                    m.store(mid, "+FLAGS", "\\Deleted")
                m.expunge()
                return len(id_list)
        return await asyncio.to_thread(_call)

    async def forward_email(
        self,
        *,
        user: str,
        message_id: str,
        to: str | list[str],
    ) -> dict[str, Any]:
        """Pull `message_id` from `user`'s inbox and resend it to `to`."""
        def _fetch() -> tuple[bytes, str, str]:
            with self._imap_factory(self._imap_host, self._imap_port) as m:
                addr = self._addr_for(user)
                m.login(addr, addr)
                m.select("INBOX")
                typ, data = m.fetch(message_id.encode("ascii"), "(RFC822)")
                if typ != "OK" or not data or not data[0]:
                    raise RuntimeError(f"message_id {message_id!r} not found")
                raw = data[0][1] if isinstance(data[0], tuple) else data[0]
                parsed = email.message_from_bytes(raw)
                return raw, parsed.get("Subject", ""), parsed.get_payload(decode=False) or ""

        raw, subj, body = await asyncio.to_thread(_fetch)
        # Resend as a fresh message tagged "Fwd:" — original message_id stays in headers.
        return await self.send_email(
            from_user=user,
            to=to,
            subject=f"Fwd: {subj}",
            body=body if isinstance(body, str) else str(body),
        )

    async def reply_email(
        self,
        *,
        user: str,
        message_id: str,
        body: str = "",
        content: str | None = None,
        text: str | None = None,
        **_extra: Any,
    ) -> dict[str, Any]:
        """Reply to an inbound message: pull `message_id` from `user`'s inbox and
        send a threaded reply (Re:/In-Reply-To) back to its original sender."""
        body = body or content or text or ""

        def _fetch() -> tuple[str, str, str]:
            with self._imap_factory(self._imap_host, self._imap_port) as m:
                addr = self._addr_for(user)
                m.login(addr, addr)
                m.select("INBOX")
                typ, data = m.fetch(message_id.encode("ascii"), "(RFC822)")
                if typ != "OK" or not data or not data[0]:
                    raise RuntimeError(f"message_id {message_id!r} not found")
                raw = data[0][1] if isinstance(data[0], tuple) else data[0]
                parsed = email.message_from_bytes(raw)
                return (parsed.get("From", ""), parsed.get("Subject", ""),
                        parsed.get("Message-ID", ""))

        sender, subj, orig_mid = await asyncio.to_thread(_fetch)
        reply_subj = subj if subj.lower().startswith("re:") else f"Re: {subj}"
        return await self.send_email(
            from_user=user, to=sender, subject=reply_subj, body=body,
            in_reply_to=orig_mid,
        )

    # ── internals ───────────────────────────────────────────────────

    def _build_message(
        self,
        *,
        from_user: str,
        to: str | list[str],
        subject: str,
        body: str,
        cc: str | list[str] | None,
        attachments: list[dict[str, Any]] | None,
        in_reply_to: str | None = None,
    ) -> EmailMessage:
        from ...audit.collector import get_current_sim_time
        msg = EmailMessage()
        # Sanitize header values: EmailMessage rejects raw CR/LF, which arise from
        # RFC-2822 folding of long Subject / Message-ID values read back from IMAP.
        # A reply to a long-subject (deep "Re: Re: ...") or long-message-id email
        # would otherwise raise "Header values may not contain linefeed or carriage
        # return characters" and drop the reply.
        def _h(v: str) -> str:
            return str(v).replace("\r", " ").replace("\n", " ").strip()
        msg["From"] = _h(from_user)
        msg["To"] = _h(", ".join(_as_list(to)))
        if cc:
            msg["Cc"] = _h(", ".join(_as_list(cc)))
        msg["Subject"] = _h(subject)
        # Thread a reply to its parent so a back-and-forth shares a conversation
        # (agents read these back; the transcript judge sees a real thread, not
        # disjoint one-offs).
        if in_reply_to:
            _irt = _h(in_reply_to)
            msg["In-Reply-To"] = _irt
            msg["References"] = _irt
        # Stamp the Date header with sim-time (RFC-2822) so the message agents read
        # back — and the transcript judge — see the simulated day, not the SMTP
        # server's wall-clock.
        _sim_ts = get_current_sim_time()
        if _sim_ts:
            import datetime as _dt
            from email.utils import format_datetime as _fmt_dt
            msg["Date"] = _fmt_dt(_dt.datetime.fromisoformat(_sim_ts.replace("Z", "+00:00")))
        msg.set_content(body)
        # Normalize attachments: agents pass a list of dicts, but also a bare
        # string (a filename/path), a single dict, or a list of strings — coerce
        # all of these so a str item doesn't crash on .get (was: AttributeError
        # 'str' object has no attribute 'get').
        if isinstance(attachments, (str, dict)):
            attachments = [attachments]
        for att in attachments or []:
            if isinstance(att, str):
                att = {"filename": att}
            if not isinstance(att, dict):
                continue
            name = str(att.get("filename", "attachment"))
            data = att.get("content", b"")
            if isinstance(data, str):
                data = data.encode("utf-8")
            maintype = str(att.get("maintype", "application"))
            subtype = str(att.get("subtype", "octet-stream"))
            msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=name)
        return msg


# ── helpers ─────────────────────────────────────────────────────────


def _as_list(addrs: str | Iterable[str] | None) -> list[str]:
    if addrs is None:
        return []
    if isinstance(addrs, str):
        return [addrs]
    return list(addrs)


def _summarise_message(raw: bytes, *, message_id: str) -> dict[str, Any]:
    from ...audit.collector import get_current_sim_time
    parsed = email.message_from_bytes(raw)
    # Body: prefer the first text/plain part; fall back to raw payload as a string.
    body = ""
    if parsed.is_multipart():
        for part in parsed.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if isinstance(payload, bytes):
                    body = payload.decode("utf-8", errors="replace")
                else:
                    body = str(payload or "")
                break
    else:
        payload = parsed.get_payload(decode=True)
        body = (
            payload.decode("utf-8", errors="replace") if isinstance(payload, bytes)
            else str(payload or "")
        )
    return {
        "id": message_id,
        "message_id": parsed.get("Message-ID", ""),
        "from": parsed.get("From", ""),
        "to": parsed.get("To", ""),
        "cc": parsed.get("Cc", ""),
        "subject": parsed.get("Subject", ""),
        "body": body,
        # Date is set to sim-time on send; fall back to the reader's current
        # sim-time only when an inbound message lacks a Date (e.g. seeded), never
        # wall-clock.
        "ts": parsed.get("Date", "") or (get_current_sim_time() or ""),
        "bytes": len(raw),
    }
