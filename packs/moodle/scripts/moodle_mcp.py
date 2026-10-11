# -*- coding: utf-8 -*-
"""Moodle MCP server (stdio JSON-RPC 2.0, zero third-party MCP deps).

Auth chain: cached session -> login with credentials -> error.
Cache: ~/.nguruvilu/moodle_cache.json (courses + assignment items, diffable).
"""
import json, os, re, sys, time, threading, urllib.parse
from datetime import datetime, timezone

try:
    import requests
except ImportError:
    print("FATAL: requests library missing", file=sys.stderr)
    raise

BASE = "https://moodle.herzen.spb.ru"
HOME = os.path.expanduser("~")
AUTH_FILE = os.path.join(HOME, ".nguruvilu", "moodle_auth.json")
SESSION_FILE = os.path.join(HOME, ".nguruvilu", "moodle_session.json")
CACHE_FILE = os.path.join(HOME, ".nguruvilu", "moodle_cache.json")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NguruviluMoodleMCP/1.0"
MOD_PATTERNS = {
    "assign": r"/mod/assign/view\.php\?id=(\d+)",
    "quiz": r"/mod/quiz/view\.php\?id=(\d+)",
    "forum": r"/mod/forum/view\.php\?id=(\d+)",
    "workshop": r"/mod/workshop/view\.php\?id=(\d+)",
}

def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)

def strip_tags(html):
    html = re.sub(r"<script.*?</script>", " ", html, flags=re.S | re.I)
    html = re.sub(r"<style.*?</style>", " ", html, flags=re.S | re.I)
    txt = re.sub(r"<[^>]+>", " ", html)
    txt = txt.replace("&nbsp;", " ").replace("&raquo;", "»").replace("&laquo;", "«")
    txt = txt.replace("&mdash;", "—").replace("&ndash;", "–").replace("&quot;", '"')
    txt = txt.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return re.sub(r"\s+", " ", txt).strip()

class Moodle:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA})
        sess = read_json(SESSION_FILE) or {}
        for k, v in (sess.get("cookies") or {}).items():
            self.s.cookies.set(k, v, domain="moodle.herzen.spb.ru")
        self._logged = False

    # ---------- auth ----------
    def _login(self):
        auth = read_json(AUTH_FILE)
        if not auth or not auth.get("username"):
            raise RuntimeError(f"no credentials at {AUTH_FILE}; put {{'username','password'}} there")
        r = self.s.get(f"{BASE}/login/index.php", timeout=30)
        m = re.search(r'name="logintoken" value="([^"]+)"', r.text)
        if not m:
            raise RuntimeError("login page has no logintoken")
        r2 = self.s.post(f"{BASE}/login/index.php", data={
            "username": auth["username"], "password": auth["password"], "logintoken": m.group(1),
        }, timeout=30, allow_redirects=True)
        if "login/index.php" in r2.url:
            raise RuntimeError("login rejected (check credentials)")
        self._logged = True
        write_json(SESSION_FILE, {"cookies": {c.name: c.value for c in self.s.cookies}, "saved_at": now_iso()})
        return True

    def ensure(self):
        if self._logged:
            return True
        r = self.s.get(f"{BASE}/my/", timeout=30, allow_redirects=True)
        if "login/index.php" in r.url:
            self._login()
            r = self.s.get(f"{BASE}/my/", timeout=30, allow_redirects=True)
            if "login/index.php" in r.url:
                raise RuntimeError("re-login failed")
            # refresh session cookie cache
            write_json(SESSION_FILE, {"cookies": {c.name: c.value for c in self.s.cookies}, "saved_at": now_iso()})
        self._my_html = r.text
        m = re.search(r'"sesskey":"([^"]+)"', r.text)
        self.sesskey = m.group(1) if m else None
        um = re.search(r'data-usertext="([^"]+)"', r.text) or re.search(r'>([^<]*\b\d{6}\b[^<]*)<', r.text)
        self.username = um.group(1).strip() if um else (read_json(AUTH_FILE) or {}).get("username", "")
        return True

    # ---------- scan ----------
    def list_courses(self):
        self.ensure()
        courses = {}
        for m in re.finditer(r'href="(?:https://moodle\.herzen\.spb\.ru)?/course/view\.php\?id=(\d+)"[^>]*>([^<]{3,160})<', self._my_html):
            cid, name = m.group(1), strip_tags(m.group(2))
            if cid not in courses and name:
                courses[cid] = name
        return courses

    @staticmethod
    def classify_label(color_css):
        c = color_css.lower()
        m = re.search(r'#([0-9a-f]{6}|[0-9a-f]{3})', c)
        if not m:
            if "rgb" in c:
                nums = [int(x) for x in re.findall(r'\d+', c)[:3]]
                if len(nums) == 3:
                    r_, g_, b_ = nums
                    if r_ > 150 and g_ < 90 and b_ < 90:
                        return "red"
                    if r_ > 130 and b_ > 130 and g_ < 110:
                        return "purple"
            return ""
        hx = m.group(1)
        if len(hx) == 3:
            hx = "".join(ch * 2 for ch in hx)
        r_, g_, b_ = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
        if r_ > 150 and g_ < 90 and b_ < 90:
            return "red"
        if r_ > 130 and b_ > 130 and g_ < 110:
            return "purple"
        if r_ > 140 and g_ < 60 and b_ < 60:
            return "red"
        return ""

    def parse_module_labels(self, html, cid):
        """Walk the course HTML in order; every activity inherits the nearest
        preceding colored label (red/purple = teacher's 'this is homework' marker)."""
        events = []
        for m in re.finditer(r'<(\w+)([^>]*style="[^"]*(?:background-color|color)\s*:[^"]*"[^>]*)>', html):
            tag, stl = m.group(1), m.group(2)
            seg = html[m.end():m.end() + 900]
            close = seg.find(f"</{tag}>")
            if close < 0:
                continue
            txt = strip_tags(seg[:close])
            if not txt or len(txt) < 3:
                continue
            kind = self.classify_label(stl)
            if kind:
                events.append((m.start(), "label", kind, txt[:200]))
        for m in re.finditer(r'href="[^"]*mod/(assign|quiz|forum|workshop)/view\.php\?id=(\d+)"', html):
            events.append((m.start(), "mod", m.group(1), m.group(2)))
        events.sort()
        labels = {}
        last = None
        for pos, typ, a, b in events:
            if typ == "label":
                last = (a, b)
            else:
                key = f"{a}:{b}"
                if last:
                    labels[key] = {"label_color": last[0], "label_text": last[1]}
                elif key in labels:
                    pass
        return labels

    def fetch_course_modules(self, cid):
        r = self.s.get(f"{BASE}/course/view.php?id={cid}", timeout=40)
        labels = self.parse_module_labels(r.text, cid)
        out = []
        for typ, pat in MOD_PATTERNS.items():
            for aid in sorted(set(re.findall(pat, r.text))):
                out.append((typ, aid, labels.get(f"{typ}:{aid}")))
        return out

    def fetch_item(self, typ, aid, course_id, course_name):
        url = f"{BASE}/mod/{typ}/view.php?id={aid}"
        try:
            r = self.s.get(url, timeout=40)
        except Exception as e:
            return {"error": str(e)[:120], "url": url}
        h = r.text
        tm = re.search(r"<title>(.*?)</title>", h, re.S)
        title = strip_tags(tm.group(1)) if tm else aid
        title = re.sub(r"\s*[»|]\s*Moodle\s*$", "", title)
        # cut "Course: " prefix if present
        if ": " in title[:90]:
            head, tail = title.split(": ", 1)
            if len(head) < 70:
                title = tail
        info = {"type": typ, "id": aid, "url": url, "title": title[:160]}
        page_txt = strip_tags(h)
        # due date (ru + en)
        dm = re.search(r"(?:Срок сдачи|Due date|Завершение попыток|Открытие попыток|Close attempts|Close the quiz)(.{0,260})", page_txt)
        if dm:
            info["due_raw"] = dm.group(1)[:160]
            d = re.search(r"\d{1,2}\s*[а-яА-Яa-zA-Z]+\s*\d{4}(?:[^0-9]{0,20}\d{1,2}:\d{2})?", dm.group(1))
            if d:
                info["due"] = d.group(0)[:60]
        # submission/attempt state (bilingual, compact)
        st_parts = []
        for pat, cap in [
            (r"(Continue your attempt|Открыть попытку|Пройти тест|Начать попытку|Начать тест)", "attempt"),
            (r"(Attempts allowed:\s*\d+)", "attempts"),
            (r"(Grade to pass:\s*[\d.,]+\s*out of\s*[\d.,]+)", "pass"),
            (r"(Submission status|Состояние отправки)(.{0,140})", "subm"),
            (r"(\d+\s+(?:Finished|In progress|Ожидает|Не начат)[^C]{0,60})", "tries"),
            (r"(Осталось попыток:?\s*\d+|Not attempted|Не начат)", "left"),
        ]:
            m = re.search(pat, page_txt)
            if m:
                st_parts.append(re.sub(r"\s+", " ", " ".join(g for g in m.groups() if g))[:140])
        if st_parts:
            info["status"] = " | ".join(st_parts)[:260]
        # teacher attachments (pluginfile, non-theme)
        atts = sorted(set(re.findall(r'href="(' + re.escape(BASE) + r'/pluginfile\.php/\d+/mod_assign/[^"]+)"', h)))
        if not atts:
            atts = sorted(set(re.findall(r'href="(' + re.escape(BASE) + r'/pluginfile\.php/\d+/mod_[^"]+)"', h)))
        if atts:
            info["attachments"] = atts[:20]
        return info

    def scan(self, refresh=True):
        self.ensure()
        cache = read_json(CACHE_FILE) or {}
        old_items = cache.get("items") or {}
        courses = self.list_courses()
        items = {}
        order = 0
        for cid, cname in courses.items():
            try:
                mods = self.fetch_course_modules(cid)
            except Exception as e:
                mods = []
                print(f"course {cid} err: {e}", file=sys.stderr)
            for typ, aid, label in mods:
                key = f"{typ}:{aid}"
                if not refresh and key in old_items:
                    items[key] = old_items[key]
                    continue
                info = self.fetch_item(typ, aid, cid, cname)
                if label:
                    info.update(label)
                prev = old_items.get(key) or {}
                info["course_id"] = cid
                info["course"] = cname
                info["first_seen"] = prev.get("first_seen") or now_iso()[:10]
                # change detection
                sig = json.dumps({k: info.get(k) for k in ("title", "due", "status")}, ensure_ascii=False)
                info["sig"] = sig
                if prev and prev.get("sig") != sig:
                    info["changed_at"] = now_iso()[:10]
                order += 1
                info["order"] = prev.get("order", order)
                items[key] = info
                time.sleep(0.12)
        cache = {"scanned_at": now_iso(), "courses": courses, "items": items}
        write_json(CACHE_FILE, cache)
        return cache

    # ---------- submission ----------
    def find_edit_url(self, aid):
        r = self.s.get(f"{BASE}/mod/assign/view.php?id={aid}", timeout=40)
        h = r.text
        m = re.search(r'href="([^"]*submission\.php\?id=\d+[^"]*action=edit[^"]*)"', h)
        if not m:
            m = re.search(r'href="([^"]*mod/assign/submission\.php[^"]*)"', h)
        if m:
            u = m.group(1).replace("&amp;", "&")
            if u.startswith("/"):
                u = BASE + u
            return u, h
        return None, h

    def parse_edit_page(self, url):
        r = self.s.get(url, timeout=40)
        h = r.text
        sk = self.sesskey
        m = re.search(r'name="sesskey" value="([^"]+)"', h)
        if m:
            sk = m.group(1)
        repo = re.search(r'"repo_id"\s*:\s*(\d+)', h)
        env = re.search(r'"env"\s*:\s*"([^"]+)"', h)
        form = re.search(r'<form[^>]*id="editsubmissionform"[^>]*action="([^"]*)"', h)
        return {
            "url": url, "sesskey": sk,
            "repo_id": int(repo.group(1)) if repo else None,
            "env": env.group(1) if env else None,
            "form_action": form.group(1) if form else url,
            "has_editor": "editsubmissionform" in h,
            "html_len": len(h),
        }

    def upload_draft(self, path, edit, file_id=1):
        """Try Moodle ajax upload endpoints; returns draft itemid or raises."""
        fname = os.path.basename(path)
        with open(path, "rb") as f:
            data = f.read()
        endpoints = [
            f"{BASE}/lib/ajax/upload.php?sesskey={urllib.parse.quote(edit['sesskey'])}",
            f"{BASE}/repository/repository_ajax.php?action=upload",
        ]
        last = None
        for url in endpoints:
            form = {
                "repo_id": str(edit.get("repo_id") or 1),
                "sesskey": edit["sesskey"] or "",
                "title": fname,
                "env": edit.get("env") or "editor",
                "accepted_types": "",
                "maxbytes": "0",
                "areamaxbytes": "0",
            }
            try:
                r = self.s.post(url, data=form,
                                files={"file": (fname, data, "application/octet-stream")}, timeout=120)
            except Exception as e:
                last = str(e)
                continue
            if r.status_code == 200:
                try:
                    j = r.json()
                except Exception:
                    j = None
                if isinstance(j, dict):
                    itemid = None
                    if j.get("draftitemid") is not None:
                        itemid = j["draftitemid"]
                    elif isinstance(j.get("results"), list) and j["results"]:
                        itemid = j["results"][0].get("itemid") or j["results"][0].get("id")
                    elif j.get("itemid") is not None:
                        itemid = j["itemid"]
                    if itemid is not None:
                        return int(itemid)
                    last = f"no itemid in response: {json.dumps(j, ensure_ascii=False)[:200]}"
                else:
                    last = f"non-json {r.status_code}: {r.text[:120]}"
            else:
                last = f"http {r.status_code}: {r.text[:120]}"
        raise RuntimeError("upload failed: " + str(last))

    def submit_files(self, aid, paths, dry_run=True):
        edit_url, view_html = self.find_edit_url(aid)
        if not edit_url:
            return {"editable": False, "reason": "no submission edit link on view page",
                    "offline": "Разрешена только загрузка файлов почтой" in view_html}
        edit = self.parse_edit_page(edit_url)
        result = {"editable": edit["has_editor"], "edit_url": edit_url,
                  "sesskey": bool(edit["sesskey"]), "repo_id": edit["repo_id"], "dry_run": dry_run}
        if dry_run or not edit["has_editor"]:
            result["files"] = [os.path.basename(p) for p in paths]
            return result
        itemid = None
        for p in paths:
            itemid = self.upload_draft(p, edit)
        # save submission
        data = {
            "id": str(aid),
            "action": "editsubmission",
            "sesskey": edit["sesskey"] or "",
            "sesskey_hidden": edit["sesskey"] or "",
            "attachments[draft][itemid]": str(itemid),
            "attachments[draft][contextid]": "0",
            "savechanges": "Сохранить изменения",
            "submitbutton": "Отправить на проверку",
        }
        r = self.s.post(edit["form_action"] or edit_url, data=data, timeout=60, allow_redirects=True)
        h = r.text
        ok = ("проверк" in h.lower() or "submitted" in h.lower() or r.status_code == 200)
        result.update({"posted": True, "http": r.status_code,
                       "status_hint": (re.search(r"(?:Submission status|Состояние отправки)(.{0,120})", strip_tags(h)) or type("", (), {"group": lambda s, i: ""})()).group(1)[:120] if ok else "",
                       "item_id": itemid})
        return result


# ---------------- MCP plumbing ----------------
md = Moodle()
TOOLS = [
    {"name": "moodle_status",
     "description": "Check Moodle login status: who is logged in, cache freshness, counts of known courses/assignments.",
     "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "moodle_list_courses",
     "description": "List enrolled courses (id + name). Uses /my/ session.",
     "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "moodle_list_assignments",
     "description": ("Diffable assignment inventory across ALL course modules (assign/quiz/forum/workshop). "
                     "Default returns only NEW and CHANGED items vs previous scan (cheap for context). "
                     "Set refresh=true to re-crawl all courses (slow ~1-3 min) and update the cache; "
                     "set new_only=false to get the full compact list without re-crawling."),
     "inputSchema": {"type": "object", "properties": {
         "refresh": {"type": "boolean", "description": "Re-crawl all courses before answering (default false)"},
         "new_only": {"type": "boolean", "description": "true(default)=only new/changed; false=full list"},
         "course_id": {"type": "string", "description": "restrict to one course id"},
         "types": {"type": "array", "items": {"type": "string", "enum": ["assign", "quiz", "forum", "workshop"]},
                   "description": "module types filter (default all)"},
     }, "additionalProperties": False}},
    {"name": "moodle_get_assignment",
     "description": "Full detail of one assignment/quiz: title, course, due date, submission status, teacher attachments (downloadable URLs).",
     "inputSchema": {"type": "object", "properties": {
         "id": {"type": "string", "description": "module id, e.g. '914551'"},
         "type": {"type": "string", "enum": ["assign", "quiz", "forum", "workshop"], "default": "assign"},
     }, "required": ["id"], "additionalProperties": False}},
    {"name": "moodle_download",
     "description": "Download teacher attachment files of an assignment into a local folder. Returns saved file paths.",
     "inputSchema": {"type": "object", "properties": {
         "id": {"type": "string", "description": "module id"},
         "type": {"type": "string", "enum": ["assign", "quiz", "forum", "workshop"], "default": "assign"},
         "out_dir": {"type": "string", "description": "target folder; default ./moodle_downloads"},
     }, "required": ["id"], "additionalProperties": False}},
    {"name": "moodle_submit",
     "description": ("Submit files to an assign module. SAFETY: dry_run=true (default) only parses the submission "
                     "form and reports readiness WITHOUT posting. Pass dry_run=false to actually upload and submit."),
     "inputSchema": {"type": "object", "properties": {
         "id": {"type": "string", "description": "assign module id"},
         "files": {"type": "array", "items": {"type": "string"}, "description": "local file paths to upload"},
         "dry_run": {"type": "boolean", "default": True, "description": "true=simulate only (default)"},
     }, "required": ["id", "files"], "additionalProperties": False}},
]

def tool_call(name, args):
    if name == "moodle_status":
        md.ensure()
        cache = read_json(CACHE_FILE) or {}
        return {"logged_in": True, "user": getattr(md, "username", ""),
                "sesskey": bool(getattr(md, "sesskey", None)),
                "courses": len(cache.get("courses") or {}),
                "items": len(cache.get("items") or {}),
                "scanned_at": cache.get("scanned_at")}
    if name == "moodle_list_courses":
        return md.list_courses()
    if name == "moodle_list_assignments":
        refresh = bool(args.get("refresh"))
        new_only = args.get("new_only", True)
        types = set(args.get("types") or ["assign", "quiz", "forum", "workshop"])
        cid = args.get("course_id")
        cache = read_json(CACHE_FILE) or {}
        if refresh or not cache.get("items"):
            cache = md.scan(refresh=True)
        items = cache.get("items") or {}
        out, new, changed = [], [], []
        for key, it in items.items():
            if it.get("type") not in types:
                continue
            if cid and str(it.get("course_id")) != str(cid):
                continue
            row = {"k": key, "t": it.get("title", "")[:110], "c": it.get("course", "")[:60],
                   "due": it.get("due", ""), "st": (it.get("status") or "")[:90], "url": it.get("url", "")}
            out.append(row)
            if it.get("first_seen", "9999") >= cache.get("scanned_at", "")[:10]:
                pass
            if it.get("changed_at"):
                changed.append(row)
        # new = items whose first_seen equals latest scan date
        scan_day = (cache.get("scanned_at") or "")[:10]
        for key, it in items.items():
            if it.get("type") not in types:
                continue
            if cid and str(it.get("course_id")) != str(cid):
                continue
            if it.get("first_seen") == scan_day:
                row = {"k": key, "t": it.get("title", "")[:110], "c": it.get("course", "")[:60],
                       "due": it.get("due", ""), "st": (it.get("status") or "")[:90], "url": it.get("url", "")}
                if row not in new:
                    new.append(row)
        if new_only:
            return {"scanned_at": cache.get("scanned_at"), "total": len(out),
                    "new": new, "changed": changed, "homework_red_purple": homework_rows}
        return {"scanned_at": cache.get("scanned_at"), "total": len(out),
                "homework_red_purple": homework_rows, "items": out}
    if name == "moodle_get_assignment":
        typ = args.get("type") or "assign"
        aid = str(args["id"])
        cache = read_json(CACHE_FILE) or {}
        it = (cache.get("items") or {}).get(f"{typ}:{aid}")
        fresh = md.fetch_item(typ, aid, (it or {}).get("course_id", "?"), (it or {}).get("course", "?"))
        fresh["course_id"] = (it or {}).get("course_id", fresh.get("course_id"))
        fresh["course"] = (it or {}).get("course", fresh.get("course"))
        # enrich: full description text
        try:
            r = md.s.get(fresh["url"], timeout=40)
            h = r.text
            dm = re.search(r'id="region-main".*?</div>\s*</div>', h, re.S)
            desc_src = h
            m = re.search(r'(?:summary|description)[^>]*>(.*?)</div>', h, re.S)
            # prefer the no-overflow content block
            m2 = re.search(r'class="no-overflow[^"]*"[^>]*>(.*?)</div>', h, re.S)
            if m2:
                desc_src = m2.group(1)
            fresh["description"] = strip_tags(desc_src)[:2000]
        except Exception as e:
            fresh["desc_error"] = str(e)[:100]
        return fresh
    if name == "moodle_download":
        typ = args.get("type") or "assign"
        aid = str(args["id"])
        out_dir = args.get("out_dir") or os.path.join(os.getcwd(), "moodle_downloads")
        os.makedirs(out_dir, exist_ok=True)
        info = tool_call("moodle_get_assignment", {"id": aid, "type": typ})
        saved, errs = [], []
        for i, url in enumerate(info.get("attachments") or []):
            try:
                r = md.s.get(url, timeout=120, stream=True)
                cd = r.headers.get("Content-Disposition", "")
                m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd)
                if m:
                    fname = urllib.parse.unquote(m.group(1))
                else:
                    fname = os.path.basename(urllib.parse.urlparse(url).path) or f"file_{i}"
                fname = re.sub(r'[<>:"/\\|?*]', "_", fname)
                path = os.path.join(out_dir, f"{typ}{aid}_{fname}")
                with open(path, "wb") as f:
                    for chunk in r.iter_content(65536):
                        f.write(chunk)
                saved.append(path)
            except Exception as e:
                errs.append({"url": url[:80], "err": str(e)[:100]})
            time.sleep(0.15)
        return {"out_dir": out_dir, "saved": saved, "errors": errs}
    if name == "moodle_submit":
        paths = [os.path.abspath(p) for p in args.get("files") or []]
        for p in paths:
            if not os.path.isfile(p):
                raise RuntimeError(f"file not found: {p}")
        return md.submit_files(str(args["id"]), paths, dry_run=args.get("dry_run", True))
    raise RuntimeError(f"unknown tool {name}")


def send(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()

def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            continue
        mid = msg.get("id")
        method = msg.get("method")
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": msg.get("params", {}).get("protocolVersion", "2024-11-05"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "moodle-herzen", "version": "1.0.0"}}})
        elif method == "notifications/initialized":
            pass
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            params = msg.get("params") or {}
            name = params.get("name")
            args = params.get("arguments") or {}
            try:
                res = tool_call(name, args)
                text = json.dumps(res, ensure_ascii=False)
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": text}], "isError": False}})
            except Exception as e:
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": "ERROR: " + str(e)[:500]}], "isError": True}})
        elif method == "ping":
            send({"jsonrpc": "2.0", "id": mid, "result": {}})
        elif mid is not None:
            send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": "method not found"}})

if __name__ == "__main__":
    main()

