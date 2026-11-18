# hf_download.py

Downloads every file of a Hugging Face repo page into its own folder. The
download can be interrupted at any point and resumed, and every file is checked
against the checksum published by the Hub. When the download is complete, the
folder is turned into a git clone of the repo.

Needs Python 3.9 or newer and no Python packages. The git step needs `git`
(2.32 or newer); use `--nogit` if you don't want it.

## Usage

```
python3 hf_download.py https://huggingface.co/zai-org/GLM-5.3-Flash/tree/main
```

This creates `./zai-org__GLM-5.3-Flash`, downloads all files into it and then
makes the folder a git clone. After a crash, a network failure or Ctrl-C, run
the same command again: finished files are skipped and partial files continue
from the last byte on disk.

To get only the files, without git:

```
python3 hf_download.py --nogit https://huggingface.co/zai-org/GLM-5.3-Flash/tree/main
```

To turn a folder downloaded earlier (with `--nogit`, or by an older version of
this script) into a git clone, without downloading anything:

```
python3 hf_download.py --togit https://huggingface.co/zai-org/GLM-5.3-Flash/tree/main
```

The URL can be a model, dataset (`/datasets/...`) or Space (`/spaces/...`)
page. A `/tree/<revision>` URL selects a branch, tag or commit, and a
`/tree/<revision>/<subfolder>` URL downloads only that subfolder. The short
form `owner/name` also works.

| Option | Default | Meaning |
| --- | --- | --- |
| `-d`, `--dir` | `.` | Directory in which the repo's folder is created |
| `-j`, `--jobs` | `4` | Number of files downloaded in parallel |
| `--retries` | `10` | Attempts without any progress before giving up on a file |
| `--verify` | off | Re-hash files that are already complete instead of trusting the manifest |
| `--nogit` | off | Only download the files; don't make the folder a git clone |
| `--togit` | off | Download nothing; make an existing, complete download a git clone |

The exit code is 0 when everything was downloaded and verified (and, unless
`--nogit` was given, converted), 1 when some files or the git step failed, and
130 after Ctrl-C.

### Gated and private repos

Set `HF_TOKEN` to an access token, or log in once with `huggingface-cli login`
(the script reads the token that command stores). For gated repos you also need
to accept the terms on the repo's web page.

## Folder names

Each page gets a folder named `<owner>__<repo>`. A revision other than `main`
is appended as `@<revision>`, so different revisions of one repo don't
overwrite each other.

## The git clone

After a complete download the folder is set up like a `git clone` of the repo:
it has the full history, `origin` points at the Hugging Face repo, and the
branch from the URL is checked out and tracks `origin`. `git status` is clean
and `git log` works. A URL that names a tag or commit id gives a detached HEAD.

This is quick and needs almost no extra disk space. Only git's own objects are
fetched, which for a model repo means the history, the small files and the
pointers to the large files (a few megabytes). The large files are the ones
already on disk. Git would normally read and hash every one of them again to
confirm they match the commit; since the script has just verified their
checksums, it tells git so instead.

Things to know:

- **Install git-lfs before using `git pull` or `git checkout` in the folder.**
  Without it, git replaces updated large files with small pointer files. The
  script prints a reminder when git-lfs is missing.
- Running the script again on a folder that is already a clone updates the
  files first and then moves the clone to the new commit. That is an
  alternative to `git pull` that keeps the resumable, verified download.
- If the folder's git repo has commits of your own, or a different `origin`,
  the script leaves its git state alone and reports an error.
- The large files are not copied into `.git/lfs/objects`, where git-lfs keeps
  its own copies. `git status`, `git log` and `git pull` don't need them, but
  if you switch to another commit and back, git-lfs has to download the large
  files that differ between the two again.
- A download of a subfolder URL cannot become a clone and is left as plain
  files.
- `--togit` first checks that every file is present and matches the current
  commit on the Hub. If the repo has changed since the download, run the script
  without `--togit` to fetch the changes; it converts afterwards.
- For gated or private repos the token is passed to git for the fetch only and
  is not stored in the clone.

## Progress display

In a terminal, each file being downloaded has its own bar, with a total at the
bottom:

```
model.safetensors         [████████████░░░░░░░░░░░░░░░]  44.6%  233.0 MiB/522.7 MiB   56.4 MiB/s  5m avg  44.1 MiB/s  ETA 0:00:06
TOTAL 25/26 files         [█████████████████████████░░]  94.6%    5.0 GiB/5.2 GiB     79.6 MiB/s  5m avg  66.6 MiB/s  ETA 0:00:04
```

The first speed covers the last 3 seconds and the second the last 5 minutes;
the ETA is based on the second. A finished file is replaced by a
`✓ <name> (<size>, verified)` line. Files that were already complete get no
line and only count towards the total. In a terminal too narrow for the bar,
only the numbers are shown.

When the output is piped or redirected, the script prints a one-line summary
every 30 seconds instead.

## How resuming and verification work

- Data is written to `<name>.<hash-prefix>.part` and resumed with an HTTP
  `Range` request. The file gets its real name only after its checksum matched,
  so a file with its final name is always complete and verified.
- The expected checksums come from the Hub's file listing: SHA-256 for LFS
  files (the large ones) and the git blob SHA-1 for small regular files.
- A partial file that fails its checksum is deleted and that file is downloaded
  again from the start.
- The part name contains the expected hash, and the revision is pinned to one
  commit per run. If the repo changes between runs, old partials are discarded
  rather than mixed with the new version, and changed files are downloaded
  again.
- `.hfdl-manifest.json` in the download folder records each verified file's
  hash, size and modification time, so a restart doesn't re-hash hundreds of
  gigabytes. A file that is missing from the manifest, or whose size or
  modification time changed, is re-hashed. `--verify` re-hashes everything.
- Stalled connections time out after 60 seconds and are retried with
  increasing delays. A file is only given up on after `--retries` attempts in a
  row that made no progress.
- Files whose path would end up outside the download folder are refused.

Files in the download folder that are not part of the repo are left alone. In
a clone, `.hfdl-manifest.json` and `*.part` files are hidden from `git status`.
