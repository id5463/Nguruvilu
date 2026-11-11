#!/usr/bin/env python3
"""Move idle session files into `archive/`, and compress the oldest of them.

Why this exists, with the numbers it was written from:

    sessions/            28 files,  72 MB   (readable, scanned on every `ready`)
    sessions/archive/   332 files, 214 MB   (already moved once, by hand)

`JsonlStore::list()` walks every `*.jsonl` directly under `sessions/` to build
the session list — that list is rebuilt on every reconnect, every session
switch, and every new session. Moving idle files out of that directory cut the
walk from 0.36 s to 0.07 s on the machine this was measured on. Nothing was
deleted then and nothing is deleted here: `archive/` is still inside the same
store root, `ngu show <id>` still finds a file if it is named explicitly.

The second half compresses what has been archived for a long time. Session JSONL
is text with a lot of repeated schema, so gzip takes it to roughly a seventh.
Those files are historical — nothing scans them — so the decompression cost is
only paid if someone explicitly asks for that session.

Run it from cron. It prints a one-line summary and changes nothing it does not
report.
"""

import argparse
import gzip
import os
import shutil
import sys
import time

DEFAULT_ROOT = os.path.expanduser("~/.nguruvilu/sessions")

# Sessions untouched for this long stop being "the conversation someone might
# still switch to". A week is long enough to cover a working week of habit and
# short enough that the directory does not grow without bound.
IDLE_DAYS = 7

# Archived files older than this are compressed. The point is not saving space
# today — it is that the archive does not itself become the next problem.
COMPRESS_AFTER_DAYS = 30


def age_days(path, now):
    try:
        return (now - os.path.getmtime(path)) / 86400.0
    except OSError:
        return 0.0


def archive_idle(root, idle_days, dry_run):
    """Move `*.jsonl` in the root that nothing has touched for `idle_days`."""
    archive = os.path.join(root, "archive")
    moved = 0
    freed = 0
    now = time.time()
    for name in sorted(os.listdir(root)):
        if not name.endswith(".jsonl"):
            continue
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            continue
        if age_days(path, now) < idle_days:
            continue
        size = os.path.getsize(path)
        if not dry_run:
            os.makedirs(archive, exist_ok=True)
            shutil.move(path, os.path.join(archive, name))
        moved += 1
        freed += size
        print("  archived %s (%.1f MB)" % (name, size / 1048576.0))
    return moved, freed


def compress_old(root, after_days, dry_run):
    """gzip archived session files that are past their useful-for-grepping age."""
    archive = os.path.join(root, "archive")
    if not os.path.isdir(archive):
        return 0, 0, 0
    count = 0
    before = 0
    after = 0
    now = time.time()
    for name in sorted(os.listdir(archive)):
        if not name.endswith(".jsonl"):
            continue
        path = os.path.join(archive, name)
        if not os.path.isfile(path):
            continue
        if age_days(path, now) < after_days:
            continue
        size = os.path.getsize(path)
        target = path + ".gz"
        if os.path.exists(target):
            continue
        if not dry_run:
            # Written to a temporary name first: a cron job that dies half way
            # through must not leave a truncated .gz that looks like the archive.
            temp = target + ".part"
            with open(path, "rb") as source, gzip.open(temp, "wb", compresslevel=6) as sink:
                shutil.copyfileobj(source, sink)
            os.replace(temp, target)
            os.remove(path)
            compressed = os.path.getsize(target)
        else:
            compressed = size // 7
        count += 1
        before += size
        after += compressed
        print("  compressed %s (%.1f MB -> %.1f MB)"
              % (name, size / 1048576.0, compressed / 1048576.0))
    return count, before, after


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default=DEFAULT_ROOT,
                        help="session store root (default: %(default)s)")
    parser.add_argument("--idle-days", type=int, default=IDLE_DAYS,
                        help="archive sessions untouched this long (default: %(default)s)")
    parser.add_argument("--compress-after-days", type=int, default=COMPRESS_AFTER_DAYS,
                        help="gzip archived files older than this (default: %(default)s)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would happen and change nothing")
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        print("session root %s does not exist; nothing to do" % args.root)
        return 0

    started = time.time()
    print("%s session root: %s" % ("DRY RUN —" if args.dry_run else "archiving", args.root))

    moved, freed = archive_idle(args.root, args.idle_days, args.dry_run)
    count, before, after = compress_old(args.root, args.compress_after_days, args.dry_run)

    remaining = len([n for n in os.listdir(args.root) if n.endswith(".jsonl")])
    print(
        "done in %.1fs: archived %d file(s) (%.1f MB out of the scanned directory), "
        "compressed %d (%.1f MB -> %.1f MB), %d still active"
        % (time.time() - started, moved, freed / 1048576.0,
           count, before / 1048576.0, after / 1048576.0, remaining)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
