# -*- coding: utf-8 -*-
"""Mailboxes MCP server (stdio JSON-RPC 2.0, zero third-party deps).

Two real mailboxes behind one tool set:

  yandex   — l1u.shen@yandex.com, direct SMTP (smtp.yandex.ru:465) + IMAP
             (imap.yandex.ru:993). No provider footer, sends as the human.
  agentmail— mingzi5876@agent.qq.com via the agently-cli binary (OAuth device
             flow). Tencent appends an "auto-sent" footer, which lands in spam.

Auth lives in ~/.nguruvilu/mail/credentials.json:
  {"yandex": {"email": "...", "app_password": "..."},
   "agentmail": {"cli": "/home/ubuntu/.npm-global/bin/agently-cli"}}
Environment overrides: MAIL_YANDEX_EMAIL, MAIL_YANDEX_PASSWORD.
"""
import email
import email.policy
import imaplib
import json
import os
import re
import shutil
import smtplib
import ssl
import subprocess
import sys
import time
from email.header import decode_header, make_header
from email.message import EmailMessage
from email.utils import formataddr, formatdate, make_msgid

HOME = os.path.expanduser("~")
CRED = os.path.join(HOME, ".nguruvilu", "mail", "credentials.json")
DEFAULT_YANDEX = {"email": "l1u.shen@yandex.com",
                  "smtp_host": "smtp.yandex.ru", "smtp_port": 465,
                  "imap_host": "imap.yandex.ru", "imap_port": 993}
DEFAULT_AGENTMAIL_CLI = os.path.join(HOME, ".npm-global", "bin", "agently-cli")


# ---------------- config ----------------
def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}


def cfg():
    c = read_json(CRED, {})
    y = dict(DEFAULT_YANDEX)
    y.update(c.get("yandex") or {})
    if os.environ.get("MAIL_YANDEX_EMAIL"):
        y["email"] = os.environ["MAIL_YANDEX_EMAIL"]
    if os.environ.get("MAIL_YANDEX_PASSWORD"):
        y["app_password"] = os.environ["MAIL_YANDEX_PASSWORD"]
    if not y.get("app_password"):
        # legacy location written when the app password was first created
        legacy = os.path.join(HOME, ".config", "mailer", "yandex_app_pw")
        if os.path.exists(legacy):
            y["app_password"] = open(legacy, encoding="utf-8").read().strip()
    a = dict(c.get("agentmail") or {})
    if not a.get("cli"):
        a["cli"] = DEFAULT_AGENTMAIL_CLI
    if not os.path.exists(a["cli"]):
        found = shutil.which("agently-cli")
        if found:
            a["cli"] = found
    return y, a


# ---------------- helpers ----------------
def decode_hdr(value):
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value)))
    except Exception:
        return value


def body_text(msg):
    try:
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    return part.get_content()
            for part in msg.walk():
                if part.get_content_type() == "text/html":
                    return part.get_content()
            return ""
        return msg.get_content()
    except Exception:
        try:
            return msg.get_payload(decode=True).decode("utf-8", "replace")
        except Exception:
            return ""


def yandex_conn(kind):
    y, _ = cfg()
    if not y.get("app_password"):
        raise RuntimeError("no yandex app password in %s" % CRED)
    if kind == "smtp":
        s = smtplib.SMTP_SSL(y["smtp_host"], int(y["smtp_port"]), timeout=45,
                             context=ssl.create_default_context())
        s.login(y["email"], y["app_password"])
        return s
    last = None
    for _ in range(4):
        try:
            m = imaplib.IMAP4_SSL(y["imap_host"], int(y["imap_port"]), timeout=60,
                                  ssl_context=ssl.create_default_context())
            m.login(y["email"], y["app_password"])
            return m
        except Exception as e:  # yandex drops the first CAPABILITY sometimes
            last = e
            time.sleep(2)
    raise last


# ---------------- yandex tools ----------------
def y_status(_):
    y, a = cfg()
    out = {"yandex": {"email": y["email"], "has_password": bool(y.get("app_password")),
                      "cred_file": CRED}}
    try:
        yandex_conn("smtp").quit()
        out["yandex"]["smtp"] = "ok"
    except Exception as e:
        out["yandex"]["smtp"] = "%s: %s" % (type(e).__name__, e)
    try:
        m = yandex_conn("imap")
        m.select("INBOX", readonly=True)
        typ, data = m.search(None, "ALL")
        out["yandex"]["inbox_total"] = len(data[0].split()) if data and data[0] else 0
        out["yandex"]["imap"] = "ok"
        m.logout()
    except Exception as e:
        out["yandex"]["imap"] = "%s: %s" % (type(e).__name__, e)
    out["agentmail"] = {"cli": a["cli"], "exists": os.path.exists(a["cli"])}
    return {"ok": True, "data": out}


def y_send(args):
    y, _ = cfg()
    to = args.get("to")
    to = [to] if isinstance(to, str) else list(to or [])
    if not to:
        return {"ok": False, "error": "no recipient"}
    msg = EmailMessage()
    msg["From"] = formataddr((args.get("from_name") or "Лю Шэнь", y["email"]))
    msg["To"] = ", ".join(to)
    if args.get("cc"):
        msg["Cc"] = args["cc"] if isinstance(args["cc"], str) else ", ".join(args["cc"])
    msg["Subject"] = args.get("subject", "")
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain=y["email"].split("@")[-1])
    msg["Reply-To"] = y["email"]
    msg.set_content(args.get("text", ""))
    if args.get("html"):
        msg.add_alternative(args["html"], subtype="html")
    s = yandex_conn("smtp")
    try:
        s.send_message(msg)
    finally:
        try:
            s.quit()
        except Exception:
            pass
    return {"ok": True, "data": {"from": y["email"], "to": to,
                                 "subject": msg["Subject"],
                                 "message_id": msg["Message-ID"]}}


def _imap_list(args, folder, criteria, limit):
    m = yandex_conn("imap")
    try:
        m.select(folder or "INBOX", readonly=True)
        typ, data = m.search(None, *criteria)
        ids = data[0].split() if data and data[0] else []
        out = []
        for i in ids[-int(limit or 20):][::-1]:
            typ, d = m.fetch(i, "(BODY.PEEK[HEADER.FIELDS (FROM TO SUBJECT DATE)])")
            raw = b"".join(x[1] for x in d if isinstance(x, tuple)).decode("utf-8", "replace")
            kv = {}
            for line in raw.split("\r\n"):
                if ":" in line:
                    k, v = line.split(":", 1)
                    kv[k.strip().lower()] = v.strip()
            out.append({"uid": i.decode(),
                        "date": kv.get("date", ""),
                        "from": decode_hdr(kv.get("from", "")),
                        "subject": decode_hdr(kv.get("subject", ""))})
        return {"ok": True, "data": {"folder": folder or "INBOX",
                                     "matched": len(ids), "messages": out}}
    finally:
        try:
            m.logout()
        except Exception:
            pass


def y_list(args):
    return _imap_list(args, args.get("folder") or "INBOX", ("ALL",), args.get("limit"))


def y_search(args):
    q = args.get("query")
    field = (args.get("field") or "from").upper()
    if not q:
        return {"ok": False, "error": "query required"}
    if field not in ("FROM", "TO", "SUBJECT", "CC", "BCC", "TEXT", "BODY"):
        field = "FROM"
    return _imap_list(args, args.get("folder") or "INBOX", (field, q), args.get("limit"))


def y_read(args):
    m = yandex_conn("imap")
    try:
        m.select(args.get("folder") or "INBOX", readonly=True)
        typ, d = m.fetch(args["uid"].encode() if isinstance(args["uid"], str) else args["uid"],
                         "(RFC822)")
        raw = b"".join(x[1] for x in d if isinstance(x, tuple))
        msg = email.message_from_bytes(raw, policy=email.policy.default)
        return {"ok": True, "data": {
            "from": decode_hdr(msg.get("From")),
            "to": decode_hdr(msg.get("To")),
            "subject": decode_hdr(msg.get("Subject")),
            "date": msg.get("Date", ""),
            "message_id": msg.get("Message-ID", ""),
            "text": body_text(msg)[:20000]}}
    finally:
        try:
            m.logout()
        except Exception:
            pass


def y_folders(_):
    m = yandex_conn("imap")
    try:
        typ, data = m.list()
        names = [d.decode("utf-8", "replace") if isinstance(d, bytes) else str(d) for d in data]
        return {"ok": True, "data": {"folders": names}}
    finally:
        try:
            m.logout()
        except Exception:
            pass


# ---------------- agentmail (agently-cli) tools ----------------
def _cli(args):
    """Run agently-cli and return parsed JSON."""
    _, a = cfg()
    if not os.path.exists(a["cli"]):
        return {"ok": False, "error": "agently-cli not found at %s" % a["cli"]}
    p = subprocess.run([a["cli"]] + args, capture_output=True, text=True, timeout=120)
    raw = (p.stdout or "").strip()
    # the CLI prints a JSON document followed by a "tip:" line
    if raw.startswith("{"):
        try:
            return json.loads(raw[: raw.rfind("}") + 1])
        except Exception:
            pass
    return {"ok": False, "error": (p.stderr or raw)[:800]}


def a_whoami(_):
    return _cli(["+me"])


def a_status(_):
    return _cli(["auth", "status"])


def a_list(args):
    cmd = ["message", "+list", "--limit", str(args.get("limit") or 10)]
    if args.get("dir"):
        cmd += ["--dir", args["dir"]]
    return _cli(cmd)


def a_read(args):
    return _cli(["message", "+read", "--id", args["message_id"]])


def a_send(args):
    """Two-phase: without confirmation_token returns a ctk; with it, sends."""
    _, a = cfg()
    path = os.path.join(HOME, ".nguruvilu", "mail", "_outgoing.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(args.get("text", ""))
    cmd = [a["cli"], "message", "+send", "--to", args["to"],
           "--subject", args.get("subject", ""),
           "--body-file", os.path.basename(path)]
    if args.get("html"):
        with open(path + ".html", "w", encoding="utf-8") as f:
            f.write(args["html"])
        cmd += ["--body-file", os.path.basename(path) + ".html"]
    if args.get("confirmation_token"):
        cmd += ["--confirmation-token", args["confirmation_token"]]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                       cwd=os.path.dirname(path))
    raw = (p.stdout or "").strip()
    if raw.startswith("{"):
        try:
            return json.loads(raw[: raw.rfind("}") + 1])
        except Exception:
            pass
    return {"ok": False, "error": (p.stderr or raw)[:800]}


# ---------------- registry ----------------
TOOLS = {
    "yandex_status": (y_status, "Check the Yandex mailbox: SMTP/IMAP auth and inbox size.", {}),
    "yandex_send": (y_send, "Send from l1u.shen@yandex.com via SMTP. No provider footer.",
                    {"to": "recipient (comma separated ok)", "subject": "subject",
                     "text": "plain body", "html": "optional html body",
                     "cc": "optional cc", "from_name": "display name"}),
    "yandex_list": (y_list, "List the newest messages in a folder.",
                    {"folder": "mailbox folder, default INBOX", "limit": "how many (default 20)"}),
    "yandex_search": (y_search, "Search a folder by FROM / TO / SUBJECT / TEXT.",
                      {"query": "search string", "field": "FROM|TO|SUBJECT|TEXT (default FROM)",
                       "folder": "folder", "limit": "max results"}),
    "yandex_read": (y_read, "Read one message by UID (from yandex_list / yandex_search).",
                    {"uid": "IMAP uid", "folder": "folder"}),
    "yandex_folders": (y_folders, "List IMAP folders.", {}),
    "agentmail_status": (a_status, "Show the Agent Mail (agent.qq.com) OAuth status.", {}),
    "agentmail_whoami": (a_whoami, "Show the Agent Mail alias and rate limits.", {}),
    "agentmail_list": (a_list, "List Agent Mail messages.",
                       {"dir": "inbox|sent|trash", "limit": "max"}),
    "agentmail_read": (a_read, "Read one Agent Mail message.", {"message_id": "msg_…"}),
    "agentmail_send": (a_send, "Send via Agent Mail (needs confirmation token from call 1).",
                       {"to": "recipient", "subject": "subject", "text": "plain body",
                        "html": "optional html", "confirmation_token": "ctk_… from first call"}),
}


def schema_for(params):
    return {"type": "object",
            "properties": {k: {"type": "string", "description": v} for k, v in params.items()},
            "additionalProperties": True}


def handle(msg):
    mid = msg.get("id")
    method = msg.get("method")
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "mail", "version": "1.0.0"}}}
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": [
            {"name": n, "description": f[1], "inputSchema": schema_for(f[2])}
            for n, f in TOOLS.items()]}}
    if method == "tools/call":
        p = msg.get("params") or {}
        name = p.get("name")
        args = p.get("arguments") or {}
        if name not in TOOLS:
            return {"jsonrpc": "2.0", "id": mid,
                    "error": {"code": -32601, "message": "unknown tool %s" % name}}
        try:
            out = TOOLS[name][0](args)
        except Exception as e:
            out = {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False, indent=1)}],
            "isError": not out.get("ok", True)}}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": mid, "result": {}}
    return {"jsonrpc": "2.0", "id": mid,
            "error": {"code": -32601, "message": "Method not found: %s" % method}}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            continue
        resp = handle(msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
