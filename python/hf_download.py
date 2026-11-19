#!/usr/bin/env python3
"""Download every file of a Hugging Face repo page, resumably and verified.

    python3 hf_download.py https://huggingface.co/zai-org/GLM-5.3-Flash/tree/main

Each page gets its own folder (here: ./zai-org__GLM-5.3-Flash). Just run the
same command again after a crash or Ctrl-C: finished files are skipped and
partial files continue from the last byte on disk.

How it stays safe:
  * Data is streamed into "<name>.<hash-prefix>.part" and only renamed to the
    real file name after its checksum matched, so a file with its final name
    is always complete and verified.
  * Checksums come from the Hub's file listing: SHA-256 for LFS files, the git
    blob SHA-1 for small regular files.
  * The part name contains the expected hash, so if the repo is updated
    between runs a stale partial is never mixed with the new version.
  * .hfdl-manifest.json remembers verified files (hash, size, mtime) so
    restarts don't re-hash everything. Use --verify to re-hash anyway.

Once everything is downloaded, the folder is turned into a git clone of the
repo (needs git; --nogit skips this, --togit converts an earlier download).
Only the small git history is fetched for that - the large files are the ones
already on disk, and they are neither downloaded nor hashed a second time.

Apart from git, only the Python standard library is needed. For gated/private
repos set HF_TOKEN (or log in with `huggingface-cli login`).
"""

import argparse
import base64
import errno
import glob
import hashlib
import http.client
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ENDPOINT = "https://huggingface.co"
CHUNK = 1 << 20
TIMEOUT = 60  # seconds without data before a connection counts as stalled
MANIFEST = ".hfdl-manifest.json"
NOW_WINDOW = 3      # seconds of history behind the "current" speed
AVG_WINDOW = 300    # seconds of history behind the average speed
STOP = threading.Event()


class Fatal(Exception):
    """An error that retrying will not fix."""


# ---------------------------------------------------------------- HTTP

class _Redirect(urllib.request.HTTPRedirectHandler):
    """Don't leak the HF token to the CDN (which also rejects it)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        old_host = urllib.parse.urlsplit(req.full_url).netloc
        if new is not None and urllib.parse.urlsplit(newurl).netloc != old_host:
            new.headers.pop("Authorization", None)
        return new


OPENER = urllib.request.build_opener(_Redirect)


def find_token():
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(var):
            return os.environ[var].strip()
    home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    try:
        return Path(home, "token").read_text().strip() or None
    except OSError:
        return None


def open_url(url, token, offset=0):
    headers = {"User-Agent": "hf-download-script/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if offset:
        headers["Range"] = f"bytes={offset}-"
    return OPENER.open(urllib.request.Request(url, headers=headers), timeout=TIMEOUT)


def get_json(url, token):
    """GET a JSON document, retrying transient errors. Returns (data, headers)."""
    for attempt in range(6):
        try:
            with open_url(url, token) as r:
                return json.load(r), r.headers
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                raise Fatal(f"HTTP {e.code} for {url} - wrong URL, or a gated/private "
                            "repo (set HF_TOKEN and accept its terms on the website)")
            err = e
        except (OSError, http.client.HTTPException, ValueError) as e:
            err = e
        if attempt == 5:
            raise Fatal(f"could not fetch {url}: {err}")
        time.sleep(2 ** attempt)


# ---------------------------------------------------------------- repo listing

class RepoFile:
    def __init__(self, path, size, algo, digest):
        self.path, self.size, self.algo, self.digest = path, size, algo, digest

    def new_hasher(self):
        if self.algo == "sha256":
            return hashlib.sha256()
        return hashlib.sha1(b"blob %d\0" % self.size)  # git blob id


_HFS_IGNORABLE = dict.fromkeys(
    [0x200C, 0x200D, 0x200E, 0x200F, 0xFEFF, *range(0x202A, 0x202F), *range(0x206A, 0x2070)])


def is_dotgit(name):
    """True if a path component could be taken for ".git" by some filesystem.

    Covers case-insensitive names, the Unicode characters HFS+ ignores, and the
    NTFS forms (trailing dots/spaces, the short name GIT~1, and a ":stream"
    suffix as in ".git::$INDEX_ALLOCATION") - the same cases git itself
    refuses to check out.
    """
    name = name.split(":", 1)[0].translate(_HFS_IGNORABLE).lower().rstrip(" .")
    return name == ".git" or re.fullmatch(r"git~\d+", name) is not None


def parse_url(url):
    """-> (kind, repo_id, revision, subfolder)"""
    if "://" not in url:
        url = f"{ENDPOINT}/{url.lstrip('/')}"
    parts = [urllib.parse.unquote(p) for p in urllib.parse.urlsplit(url).path.split("/") if p]
    kind = "models"
    if parts and parts[0] in ("datasets", "spaces", "models"):
        kind = parts.pop(0)
    rev, sub = "main", []
    for marker in ("tree", "blob", "resolve"):
        if marker in parts[:3]:
            i = parts.index(marker)
            if len(parts) > i + 1:
                rev = parts[i + 1]
            sub = parts[i + 2:] if marker == "tree" else []
            parts = parts[:i]
            break
    if not 1 <= len(parts) <= 2 or any(p in (".", "..") for p in parts):
        raise Fatal(f"cannot find a repo name in {url!r}")
    return kind, "/".join(parts), rev, "/".join(sub)


def list_files(kind, repo, rev, sub, token):
    """Pin the revision to a commit and list its files. -> (commit, [RepoFile])"""
    q = urllib.parse.quote
    info, _ = get_json(f"{ENDPOINT}/api/{kind}/{q(repo)}/revision/{q(rev, safe='')}", token)
    commit = info["sha"]
    url = f"{ENDPOINT}/api/{kind}/{q(repo)}/tree/{commit}"
    if sub:
        url += "/" + q(sub)
    url += "?recursive=true"
    files = []
    while url:
        entries, headers = get_json(url, token)
        for e in entries:
            if e.get("type") != "file":
                continue
            path = e["path"]
            parts = path.split("/")
            if path.startswith("/") or ".." in parts or "\\" in path:
                raise Fatal(f"refusing unsafe path from server: {path!r}")
            # Control characters would be sent to the terminal as escape codes.
            if any(ord(c) < 0x20 or 0x7F <= ord(c) <= 0x9F for c in path):
                raise Fatal(f"refusing path with control characters: {path!r}")
            if any(is_dotgit(p) for p in parts):
                # A file under .git could plant git config or hooks that run
                # commands as soon as git is used in the folder.
                raise Fatal(f"refusing repo with a file inside a .git folder: {path!r}")
            lfs = e.get("lfs")
            if lfs:
                files.append(RepoFile(path, lfs.get("size", e["size"]), "sha256", lfs["oid"]))
            else:
                files.append(RepoFile(path, e["size"], "git-sha1", e["oid"]))
        m = re.search(r'<([^>]+)>;\s*rel="next"', headers.get("Link", ""))
        url = urllib.parse.urljoin(url, m.group(1)) if m else None
    return commit, files


# ---------------------------------------------------------------- local state

class Manifest:
    """Records verified files so restarts need not re-hash them."""

    def __init__(self, path):
        self.path, self.lock = path, threading.Lock()
        try:
            self.data = json.loads(path.read_text())
        except (OSError, ValueError):
            self.data = {}

    def matches(self, f, st):
        return self.data.get(f.path) == [f.digest, st.st_size, st.st_mtime_ns]

    def record(self, f, st):
        with self.lock:
            self.data[f.path] = [f.digest, st.st_size, st.st_mtime_ns]
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(self.data, indent=0))
            os.replace(tmp, self.path)


class Rate:
    """Network bytes over time, for a current and a 5-minute average speed."""

    def __init__(self):
        self.fetched = 0
        self.samples = deque([(time.monotonic(), 0)])   # (time, fetched so far)

    def sample(self, now):
        if now - self.samples[-1][0] >= 0.5:
            self.samples.append((now, self.fetched))
            while len(self.samples) > 2 and self.samples[1][0] <= now - AVG_WINDOW:
                self.samples.popleft()

    def speed(self, now, window):
        """Bytes/s over the last `window` seconds (or since the start, if shorter)."""
        t0, n0 = self.samples[0]
        for t, n in reversed(self.samples):
            t0, n0 = t, n
            if t <= now - window:
                break
        return (self.fetched - n0) / max(now - t0, 1e-6)


class Progress:
    """Live display: one bar per file being downloaded plus a total bar."""

    def __init__(self, total_bytes, total_files):
        self.total, self.nfiles = total_bytes, total_files
        self.done = self.files_done = 0
        self.rate = Rate()
        self.active = {}    # path -> [bytes on disk, size, Rate]
        self.drawn = 0      # lines of the live region currently on screen
        self.lock = threading.Lock()
        self.tty = sys.stderr.isatty()
        self.start = self.last = time.monotonic()

    def begin(self, f):
        with self.lock:
            self.active[f.path] = [0, f.size, Rate()]

    def add(self, n, network=False, path=None):
        with self.lock:
            self.done += n
            if network:
                self.rate.fetched += n
            entry = self.active.get(path)
            if entry:
                entry[0] += n
                if network:
                    entry[2].fetched += n

    def end(self, f, result):
        with self.lock:
            was_active = self.active.pop(f.path, None)
            if result in ("ok", "skipped"):
                self.files_done += 1
            if was_active and result == "ok":
                self._println(f"\u2713 {f.path}  ({human(f.size)}, verified)")

    def log(self, msg):
        with self.lock:
            self._println(msg)

    def show(self, force=False):
        now = time.monotonic()
        if not force and now - self.last < (0.2 if self.tty else 30):
            return
        self.last = now
        with self.lock:
            self.rate.sample(now)
            for entry in self.active.values():
                entry[2].sample(now)
            if self.tty:
                self._render()
            else:
                pct = 100 * self.done / self.total if self.total else 100
                print(f"{self.files_done}/{self.nfiles} files  {human(self.done)}/"
                      f"{human(self.total)}  {pct:5.1f}%  "
                      f"{human(self.rate.speed(now, NOW_WINDOW))}/s  "
                      f"5m avg {human(self.rate.speed(now, AVG_WINDOW))}/s",
                      file=sys.stderr, flush=True)

    # The methods below expect self.lock to be held.

    def _println(self, msg):
        self._render(msg)

    def _render(self, msg=None):
        """Redraw the live region, optionally printing a permanent line above it.

        The new frame overwrites the old one in place and goes out in a single
        write. Erasing the region first would leave it blank for a moment,
        which shows up as flashing.
        """
        if not self.tty:
            if msg is not None:
                sys.stderr.write(msg + "\n")
                sys.stderr.flush()
            return
        cols, rows = shutil.get_terminal_size()
        now = time.monotonic()
        lines = [bar_line(path, done, size, rate, now, cols - 1)
                 for path, (done, size, rate) in list(self.active.items())[:max(rows - 2, 1)]]
        lines.append(bar_line(f"TOTAL {self.files_done}/{self.nfiles} files", self.done, self.total,
                              self.rate, now, cols - 1))
        out = []
        if self.drawn:      # back to the first line of the old frame
            out.append("\r" + (f"\033[{self.drawn - 1}A" if self.drawn > 1 else ""))
        if msg is not None:
            out.append(msg + "\033[K\n")
        # \033[K clears the rest of each line, \033[J whatever is left below.
        out.append("\033[K\n".join(lines) + "\033[K\033[J")
        sys.stderr.write("".join(out))
        sys.stderr.flush()
        self.drawn = len(lines)


def bar_line(label, done, size, rate, now, width):
    cur, avg = rate.speed(now, NOW_WINDOW), rate.speed(now, AVG_WINDOW)
    frac = min(done / size, 1) if size else 1
    if done >= size:
        eta = "0:00:00"
    elif avg > 0:   # the average is steadier than the current speed
        secs = int((size - done) / avg)
        eta = f"{secs // 3600}:{secs % 3600 // 60:02}:{secs % 60:02}"
    else:
        eta = "-:--:--"
    tail = (f" {frac * 100:5.1f}%  {human(done):>9}/{human(size):<9}"
            f"  {human(cur):>9}/s  5m avg {human(avg):>9}/s  ETA {eta}")
    name_w = max(12, min(40, width // 5))
    bar_w = width - name_w - len(tail) - 3
    if bar_w < 5:   # narrow terminal: drop the bar, keep the numbers
        name_w = max(8, width - len(tail))
    if len(label) > name_w:
        label = "\u2026" + label[-(name_w - 1):]
    if bar_w < 5:
        return (label.ljust(name_w) + tail)[:width]
    fill = int(frac * bar_w)
    return f"{label:<{name_w}} [{'\u2588' * fill}{'\u2591' * (bar_w - fill)}]{tail}"


def human(n):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def hash_file(path, f):
    h = f.new_hasher()
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK):
            if STOP.is_set():
                break
            h.update(chunk)
    return h


# ---------------------------------------------------------------- download

def dest_path(root, f):
    """Where f belongs on disk; refuses anything that would land outside root."""
    dest = root / f.path
    # Belt and braces on top of the listing check: also catches symlinks that
    # lead out of the folder and Windows drive-style names like "C:x".
    if not dest.resolve().is_relative_to(root):
        raise Fatal(f"{f.path}: would be written outside {root}, skipping")
    return dest


def already_complete(f, dest, manifest, verify):
    """True if dest is a verified copy of f. Hashes it unless the manifest vouches for it."""
    if not dest.is_file():
        return False
    st = dest.stat()
    if st.st_size != f.size:
        return False
    if not verify and manifest.matches(f, st):
        return True
    if hash_file(dest, f).hexdigest() == f.digest and not STOP.is_set():
        manifest.record(f, st)
        return True
    return False


def download_file(f, url, root, token, manifest, progress, verify, retries):
    """Fetch one file. Returns 'ok', 'skipped' or 'interrupted'; raises Fatal."""
    dest = dest_path(root, f)
    part = dest.with_name(f"{dest.name}.{f.digest[:12]}.part")
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Partials belonging to an older version of this file are useless now.
    for stale in glob.glob(glob.escape(str(dest)) + ".*.part"):
        if Path(stale) != part:
            os.unlink(stale)

    if already_complete(f, dest, manifest, verify):
        progress.add(f.size)
        return "skipped"
    if STOP.is_set():
        return "interrupted"
    if dest.exists():
        progress.log(f"{f.path}: existing file is corrupt or outdated, re-downloading")
        dest.unlink()

    progress.begin(f)
    hasher, hashed = None, -1   # hasher state covers the first `hashed` bytes of part
    counted = 0                 # bytes of this file already shown in progress
    failures = 0
    while True:
        if STOP.is_set():
            return "interrupted"
        have = part.stat().st_size if part.exists() else 0
        if have > f.size:       # cannot be a prefix of the right file
            part.unlink()
            have = 0
        if hashed != have:      # first pass / after an error: re-sync hash with disk
            hasher = hash_file(part, f) if have else f.new_hasher()
            hashed = have
            if STOP.is_set():
                return "interrupted"
        progress.add(have - counted, path=f.path)
        counted = have
        before = have

        try:
            if f.size == 0:
                part.touch()
            elif have < f.size:
                with open_url(url, token, offset=have) as r:
                    if have and r.status != 206:    # server ignored Range: start over
                        progress.add(-counted, path=f.path)
                        have = counted = hashed = 0
                        hasher = f.new_hasher()
                    with open(part, "ab" if have else "wb") as out:
                        while chunk := r.read(CHUNK):
                            if STOP.is_set():
                                return "interrupted"
                            out.write(chunk)
                            hasher.update(chunk)
                            hashed += len(chunk)
                            counted += len(chunk)
                            progress.add(len(chunk), network=True, path=f.path)
                        out.flush()
                        os.fsync(out.fileno())
            error = None
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                raise Fatal(f"{f.path}: HTTP {e.code} (gated/private repo? set HF_TOKEN)")
            if e.code == 416:
                part.unlink(missing_ok=True)
            error = e
        except (OSError, http.client.HTTPException) as e:
            if isinstance(e, OSError) and e.errno in (errno.ENOSPC, errno.EDQUOT, errno.EROFS):
                raise Fatal(f"{f.path}: {e}")
            error = e

        size = part.stat().st_size if part.exists() else 0
        if size == f.size and hashed == size:
            if hasher.hexdigest() == f.digest:
                os.replace(part, dest)
                manifest.record(f, dest.stat())
                return "ok"
            error = "checksum mismatch, discarding the downloaded data"
            part.unlink()
        elif error is None and size >= f.size:
            continue  # sizes disagree with our bookkeeping: loop re-syncs and re-checks

        failures = 0 if size > before else failures + 1
        if failures > retries:
            raise Fatal(f"{f.path}: giving up after {retries} retries ({error})")
        if error is not None:
            delay = min(60, 2 ** failures)
            progress.log(f"{f.path}: {error} - retrying in {delay}s")
            STOP.wait(delay)


# ---------------------------------------------------------------- git

def git_convert(root, kind, repo, rev, commit, token):
    """Turn the fully downloaded folder into a git clone checked out at `commit`.

    Only git's own objects are fetched (history, small files, LFS pointers).
    Every file on disk has just been verified against that commit, so the
    index is filled in without making git re-read the large files.
    """
    git = shutil.which("git")
    if not git:
        raise Fatal("git is not installed")
    if not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        raise Fatal(f"unexpected commit id from server: {commit!r}")
    url = f"{ENDPOINT}/{'' if kind == 'models' else kind + '/'}{repo}"
    say = lambda msg: print(msg, file=sys.stderr, flush=True)

    def with_config(base, pairs):
        """Copy of environment `base` that passes extra `git -c key=value` settings."""
        n = int(base.get("GIT_CONFIG_COUNT", "0") or 0)
        out = dict(base, GIT_CONFIG_COUNT=str(n + len(pairs)))
        for i, (key, value) in enumerate(pairs, n):
            out[f"GIT_CONFIG_KEY_{i}"], out[f"GIT_CONFIG_VALUE_{i}"] = key, value
        return out

    # No hooks and no fsmonitor command for the git calls made here, whatever a
    # .git folder that is already present may say.
    env = with_config(dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_LFS_SKIP_SMUDGE="1"),
                      [("core.hooksPath", os.devnull), ("core.fsmonitor", "false")])
    auth_env = env
    if token:   # for the fetch only, via the environment so it is not visible in `ps`
        basic = base64.b64encode(f"hf_user:{token}".encode()).decode()
        auth_env = with_config(env, [(f"http.{ENDPOINT}/.extraHeader",
                                      f"Authorization: Basic {basic}")])

    def run(*cmd, check=True, env=env, config=()):
        full = [git] + [x for kv in config for x in ("-c", kv)] + list(cmd)
        r = subprocess.run(full, cwd=root, env=env, capture_output=True, text=True)
        if check and r.returncode:
            raise Fatal(f"git {cmd[0]} failed: {(r.stderr or r.stdout).strip()}")
        return r

    say(f"turning {root.name} into a git clone of {url} ...")
    if not (root / ".git").exists():    # checked by path: a parent folder may be a repo too
        run("init", "-q")
    origin = run("remote", "get-url", "origin", check=False)
    if origin.returncode:
        run("remote", "add", "origin", url)
    elif origin.stdout.strip().rstrip("/").removesuffix(".git") != url:
        raise Fatal(f"{root} is already a git repo with another origin "
                    f"({origin.stdout.strip()}); leaving it untouched")

    run("fetch", "-q", "origin", env=auth_env)
    if run("cat-file", "-e", commit + "^{commit}", check=False).returncode:
        run("fetch", "-q", "origin", commit, env=auth_env)     # commit not on any branch/tag

    head = run("rev-parse", "-q", "--verify", "HEAD", check=False).stdout.strip()
    if not head:
        # New repo. If the page was a branch, check that branch out and make it
        # track origin; for a tag or commit id, leave a detached HEAD.
        if run("rev-parse", "-q", "--verify", f"refs/remotes/origin/{rev}", check=False).returncode == 0:
            run("symbolic-ref", "HEAD", f"refs/heads/{rev}")
            run("update-ref", "HEAD", commit)
            run("config", f"branch.{rev}.remote", "origin")
            run("config", f"branch.{rev}.merge", f"refs/heads/{rev}")
        else:
            run("update-ref", "--no-deref", "HEAD", commit)
    elif head != commit:
        # The folder was a clone already and the repo has moved on since.
        if run("merge-base", "--is-ancestor", head, commit, check=False).returncode:
            raise Fatal(f"the git repo in {root} has commits that are not in {commit[:12]}; "
                        "leaving its git state untouched")
        run("update-ref", "HEAD", commit)

    run("remote", "set-head", "origin", "--auto", check=False, env=auth_env)  # origin/HEAD, as clone does

    # Point the index at the commit without touching the files on disk.
    run("read-tree", commit)

    exclude = root / ".git" / "info" / "exclude"
    exclude.parent.mkdir(exist_ok=True)
    have = exclude.read_text() if exclude.exists() else ""
    missing = [pat for pat in (MANIFEST, MANIFEST + ".tmp", "*.part") if pat not in have.split("\n")]
    if missing:
        exclude.write_text(have + ("" if have.endswith("\n") or not have else "\n")
                           + "\n".join(missing) + "\n")

    # Let git record each file as unchanged. Normally it would push every LFS
    # file through `git lfs clean`, i.e. read and hash all of it again (and
    # git-lfs would keep a second copy under .git/lfs). The checksums were
    # verified moments ago, so for this one command the LFS filter is replaced
    # by one that answers with the pointer stored in the commit. Small non-LFS
    # files are still hashed by git itself. The user's global/system git config
    # is hidden for this command, because git-lfs registers a filter there that
    # would take precedence.
    time.sleep(1.1)     # files as new as the index would be re-checked by git later
    bare_env = dict(env, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull,
                    GIT_CONFIG_NOSYSTEM="1")
    run("update-index", "-q", "--refresh", check=False, env=bare_env,
        config=(f"filter.lfs.clean=git cat-file blob {commit}:%f", "filter.lfs.required=false"))

    status = run("status", "--porcelain").stdout.splitlines()
    branch = run("symbolic-ref", "-q", "--short", "HEAD", check=False).stdout.strip()
    say(f"git clone ready: {'branch ' + branch if branch else 'detached HEAD'} at {commit[:12]}")
    if status:
        say(f"note: git reports {len(status)} difference(s) from that commit:")
        for line in status[:10]:
            say("  " + line)
    if run("lfs", "version", check=False).returncode:
        say("note: git-lfs is not installed. The folder is a valid clone, but install git-lfs "
            "before running\n      git pull/checkout in it, or large files will be replaced "
            "by small pointer files.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("url", help="Hugging Face page URL, e.g. https://huggingface.co/org/name/tree/main")
    ap.add_argument("-d", "--dir", default=".", help="where to create the repo's folder (default: .)")
    ap.add_argument("-j", "--jobs", type=int, default=4, help="parallel downloads (default: 4)")
    ap.add_argument("--retries", type=int, default=10,
                    help="consecutive attempts without any progress before giving up on a file")
    ap.add_argument("--verify", action="store_true",
                    help="re-hash files that are already complete instead of trusting the manifest")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--nogit", action="store_true",
                      help="only download the files; don't turn the folder into a git clone")
    mode.add_argument("--togit", action="store_true",
                      help="download nothing; turn an existing, complete download into a git clone")
    args = ap.parse_args()

    token = find_token()
    kind, repo, rev, sub = parse_url(args.url)
    commit, files = list_files(kind, repo, rev, sub, token)
    if not files:
        raise Fatal("no files found on that page")

    folder = repo.replace("/", "__") + ("" if rev == "main" else f"@{rev.replace('/', '_')}")
    root = Path(args.dir, folder).resolve()
    if args.togit:
        if sub:
            raise Fatal("a subfolder download cannot become a git clone; use the repo's main page URL")
        if not root.is_dir():
            raise Fatal(f"{root} does not exist, so there is nothing to convert")
        manifest = Manifest(root / MANIFEST)
        print(f"checking {len(files)} files in {root} ...", file=sys.stderr)
        bad = [f.path for f in files if not already_complete(f, dest_path(root, f), manifest, args.verify)]
        if bad:
            raise Fatal(f"{len(bad)} of {len(files)} files are missing, incomplete or outdated "
                        f"(e.g. {bad[0]}); run without --togit to finish the download first")
        git_convert(root, kind, repo, rev, commit, token)
        return 0

    root.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(root / MANIFEST)
    progress = Progress(sum(f.size for f in files), len(files))
    print(f"{repo} @ {commit[:12]}: {len(files)} files, {human(progress.total)} -> {root}",
          file=sys.stderr)

    prefix = "" if kind == "models" else kind + "/"
    base = f"{ENDPOINT}/{prefix}{urllib.parse.quote(repo)}/resolve/{commit}/"
    failed, interrupted = [], False

    def job(f):
        result = "failed"
        try:
            result = download_file(f, base + urllib.parse.quote(f.path), root, token,
                                   manifest, progress, args.verify, args.retries)
        except Fatal as e:
            failed.append(str(e))
            progress.log(f"FAILED {e}")
        except Exception as e:  # keep the other downloads going
            failed.append(f"{f.path}: {e!r}")
            progress.log(f"FAILED {f.path}: {e!r}")
        finally:
            progress.end(f, result)

    pool = ThreadPoolExecutor(max_workers=max(1, args.jobs))
    # Small files first: the folder becomes usable/inspectable quickly.
    futures = [pool.submit(job, f) for f in sorted(files, key=lambda f: f.size)]
    try:
        while not all(fut.done() for fut in futures):
            progress.show()
            time.sleep(0.2)
    except KeyboardInterrupt:
        interrupted = True
        STOP.set()
        progress.log("interrupted - partial files are kept, run the same command to resume")
    pool.shutdown(wait=True, cancel_futures=True)
    progress.show(force=True)
    print(file=sys.stderr)

    if interrupted:
        return 130
    if failed:
        print(f"{len(failed)} file(s) failed; run the same command again to retry:", file=sys.stderr)
        for msg in failed:
            print("  " + msg, file=sys.stderr)
        return 1
    print(f"done: all {len(files)} files downloaded and verified in {root}", file=sys.stderr)
    if args.nogit:
        return 0
    if sub:
        print("note: only a subfolder was downloaded, so it was not turned into a git clone",
              file=sys.stderr)
        return 0
    try:
        git_convert(root, kind, repo, rev, commit, token)
    except Fatal as e:
        print(f"error: the files are downloaded and verified, but making the folder a git clone "
              f"failed:\n  {e}\nRun again with --togit to retry, or with --nogit to skip this step.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Fatal as e:
        sys.exit(f"error: {e}")
    except KeyboardInterrupt:
        sys.exit(130)
