# hf_download.py

Downloads every file of a Hugging Face repo page into its own folder. The
download can be interrupted at any point and resumed, and every file is checked
against the checksum published by the Hub.

Needs Python 3.9 or newer; no packages to install.

## Usage

```
python3 hf_download.py https://huggingface.co/zai-org/GLM-5.3-Flash/tree/main
```

This creates `./zai-org__GLM-5.3-Flash` and downloads all files into it. After
a crash, a network failure or Ctrl-C, run the same command again: finished
files are skipped and partial files continue from the last byte on disk.

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

The exit code is 0 when everything was downloaded and verified, 1 when some
files failed, and 130 after Ctrl-C.

### Gated and private repos

Set `HF_TOKEN` to an access token, or log in once with `huggingface-cli login`
(the script reads the token that command stores). For gated repos you also need
to accept the terms on the repo's web page.

## Folder names

Each page gets a folder named `<owner>__<repo>`. A revision other than `main`
is appended as `@<revision>`, so different revisions of one repo don't
overwrite each other.

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

Files in the download folder that are not part of the repo are left alone.
