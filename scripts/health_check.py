#!/usr/bin/env python3
"""Report the things that go wrong silently on a small machine, and clean `/tmp`.

Each check here exists because one of them already failed quietly on this host:

  disk    reached 86% while nobody was looking. The failure mode it produces is
          not "a write fails" — it is a session file that loses content, which
          reads as the agent forgetting things.
  memory  1.9 GB total. A headless browser is ~1 GB. When swap runs out the
          kernel starts killing, and the first casualty is usually sshd.
  /tmp    grew to 3.7 GB from leftover Chrome profile directories, each a few
          hundred MB. None of it was needed; all of it was invisible.

`/tmp` cleanup is restricted to entries older than `--tmp-days` and only ever
touches what is *inside* /tmp, so a cron job cannot wander into anything else.
Nothing outside /tmp is deleted by this script, ever.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

# What counts as "worth telling someone about". Both are percentages of total.
DISK_WARN = 85
MEM_WARN = 90


def disk(path="/"):
    usage = shutil.disk_usage(path)
    percent = usage.used * 100.0 / usage.total
    return {
        "path": path,
        "total_gb": usage.total / 1073741824.0,
        "free_gb": usage.free / 1073741824.0,
        "percent": percent,
        "alarm": percent >= DISK_WARN,
    }


def memory():
    """Used memory as a percentage, swap included.

    Swap matters more than RAM here: the failure that hurts is "everything is in
    swap and nothing can be paged in", which reads as a hung machine while free
    RAM still looks fine.
    """
    info = {}
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                info[key.strip()] = int(rest.split()[0]) * 1024
    except OSError:
        return None
    total = info.get("MemTotal", 0)
    available = info.get("MemAvailable", info.get("MemFree", 0))
    swap_total = info.get("SwapTotal", 0)
    swap_free = info.get("SwapFree", 0)
    used = total - available
    return {
        "total_gb": total / 1073741824.0,
        "used_gb": used / 1073741824.0,
        "percent": used * 100.0 / total if total else 0.0,
        "swap_used_gb": (swap_total - swap_free) / 1073741824.0,
        "swap_total_gb": swap_total / 1073741824.0,
    }


def tmp_size(path="/tmp"):
    total = 0
    biggest = []
    try:
        for name in os.listdir(path):
            full = os.path.join(path, name)
            try:
                if os.path.islink(full):
                    continue
                if os.path.isdir(full):
                    size = sum(
                        os.path.getsize(os.path.join(dirpath, f))
                        for dirpath, _, files in os.walk(full)
                        for f in files
                        if os.path.exists(os.path.join(dirpath, f))
                    )
                else:
                    size = os.path.getsize(full)
            except OSError:
                continue
            total += size
            biggest.append((size, name))
    except OSError:
        return {"path": path, "total_mb": 0, "biggest": []}
    biggest.sort(reverse=True)
    return {
        "path": path,
        "total_mb": total / 1048576.0,
        "biggest": [(n, s / 1048576.0) for s, n in biggest[:5]],
    }


def clean_tmp(path="/tmp", older_than_days=3, dry_run=True):
    """Remove /tmp entries untouched for `older_than_days`.

    Deliberately conservative: only direct children of `path`, never following a
    symlink, and never acting on something modified recently. A dry run is the
    default because a cleanup that surprises someone is worse than a full disk
    they were told about.
    """
    removed = []
    freed = 0
    now = time.time()
    try:
        names = os.listdir(path)
    except OSError:
        return removed, freed
    for name in names:
        full = os.path.join(path, name)
        try:
            if os.path.islink(full):
                # A symlink is one entry; deleting it costs nothing and could
                # point anywhere, so it is left alone.
                continue
            stat = os.lstat(full)
            if (now - stat.st_mtime) < older_than_days * 86400:
                continue
            size = os.path.getsize(full) if os.path.isfile(full) else 0
            if os.path.isdir(full):
                size = sum(
                    os.path.getsize(os.path.join(d, f))
                    for d, _, files in os.walk(full)
                    for f in files
                    if os.path.exists(os.path.join(d, f))
                )
            if not dry_run:
                if os.path.isdir(full):
                    shutil.rmtree(full, ignore_errors=True)
                else:
                    try:
                        os.remove(full)
                    except OSError:
                        continue
            removed.append((name, size / 1048576.0))
            freed += size
        except OSError:
            continue
    return removed, freed


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--clean-tmp", action="store_true",
                        help="actually remove old /tmp entries (default: only report)")
    parser.add_argument("--tmp-days", type=int, default=3,
                        help="only touch /tmp entries older than this (default: %(default)s)")
    parser.add_argument("--quiet", action="store_true",
                        help="print nothing unless something is in alarm")
    args = parser.parse_args()

    problems = []
    lines = []

    disk_info = disk("/")
    lines.append("disk  %s %.1f/%.1f GB used (%.0f%%)%s"
                 % (disk_info["path"],
                    disk_info["total_gb"] - disk_info["free_gb"],
                    disk_info["total_gb"], disk_info["percent"],
                    "  ** ALARM **" if disk_info["alarm"] else ""))
    if disk_info["alarm"]:
        problems.append("disk %.0f%% full, %.1f GB left"
                        % (disk_info["percent"], disk_info["free_gb"]))

    mem = memory()
    if mem:
        alarm = mem["percent"] >= MEM_WARN
        lines.append("mem   %.1f/%.1f GB used (%.0f%%), swap %.1f/%.1f GB%s"
                     % (mem["used_gb"], mem["total_gb"], mem["percent"],
                        mem["swap_used_gb"], mem["swap_total_gb"],
                        "  ** ALARM **" if alarm else ""))
        if alarm:
            problems.append("memory %.0f%% used" % mem["percent"])

    tmp = tmp_size("/tmp")
    lines.append("tmp   %.0f MB under %s" % (tmp["total_mb"], tmp["path"]))
    for name, size in tmp["biggest"]:
        lines.append("        %8.1f MB  %s" % (size, name))

    if args.clean_tmp:
        removed, freed = clean_tmp("/tmp", args.tmp_days, dry_run=False)
        lines.append("tmp   removed %d entr(ies), %.0f MB freed"
                     % (len(removed), freed / 1048576.0))
        for name, size in removed[:10]:
            lines.append("        %8.1f MB  %s" % (size, name))
    else:
        would, freed = clean_tmp("/tmp", args.tmp_days, dry_run=True)
        if would:
            lines.append("tmp   %d entr(ies) older than %d days (%.0f MB) would be removed "
                         "with --clean-tmp"
                         % (len(would), args.tmp_days, freed / 1048576.0))

    if not args.quiet or problems:
        for line in lines:
            print(line)
        if problems:
            print("ALARM: " + "; ".join(problems))
        else:
            print("OK: nothing above threshold (disk < %d%%, memory < %d%%)"
                  % (DISK_WARN, MEM_WARN))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
