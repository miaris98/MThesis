#!/usr/bin/env python3
"""Push a folder (or a list of files) to a Google Cloud Storage bucket, with an MD5 check after every upload.

Third stage of the sync pipeline (2026-10-19): E:\\MThesis_EXP first, then Google Cloud Storage (Hugging Face's private
storage is full). Run it ON THE BOX that holds the files, or on the PC for a folder of E:. Idempotent and resumable:
a file that already exists in the bucket with the same size and MD5 is skipped, so a second run after a broken
connection only sends what is missing. Large files go through GCS resumable uploads (chunks of 32 MiB).

    python scripts/sync/gcs_push.py --bucket MY_BUCKET --prefix atari/q_screen_2026-10-19/A1 --dir /workspace/q/eval
    python scripts/sync/gcs_push.py --bucket MY_BUCKET --prefix ckpt --files /workspace/ckpt_in/a.pt /workspace/ckpt_in/b.pt

Credentials (the user's, never read from dotfiles by this script): `--credentials PATH` (a service-account JSON that
the user uploaded, e.g. to /dev/shm/gcs_key.json on the box), else `$GOOGLE_APPLICATION_CREDENTIALS`, else
`--token-file PATH` (a short-lived OAuth access token, e.g. from `gcloud auth print-access-token`), else the machine's
application-default credentials. The last line printed is `GCS_OK uploaded=N skipped=M bytes=B` or `GCS_FAILED ...`
(exit code 1), so a shell loop can check it. In-progress files (`*.tmp`, `*.tmp.*`) are never uploaded.
"""
import argparse
import base64
import fnmatch
import hashlib
import os
import sys
import time
from pathlib import Path

#: Same rule as hf_push_checkpoints.py: the trainers write `<file>.tmp` and then rename it.
IN_PROGRESS_PATTERNS = ("*.tmp", "*.tmp.*", "*.tmp[0-9]*")
CHUNK = 32 * 1024 * 1024   # a multiple of 256 KiB, as GCS requires for resumable chunks


def is_in_progress(name: str) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in IN_PROGRESS_PATTERNS)


def md5_b64(path: Path) -> str:
    """Base64 MD5 of a file, the format GCS reports in `blob.md5_hash`."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(block)
    return base64.b64encode(h.digest()).decode("ascii")


def collect(dir_path, files):
    """[(local_path, relative_name)]: every file under `dir_path` (relative names keep the sub-folders) and every explicit file."""
    out = []
    if dir_path:
        root = Path(dir_path)
        for p in sorted(root.rglob("*")):
            if p.is_file() and not is_in_progress(p.name):
                out.append((p, p.relative_to(root).as_posix()))
    for f in files or []:
        p = Path(f)
        if p.is_file() and not is_in_progress(p.name):
            out.append((p, p.name))
    return out


def push(bucket, prefix, items, retries=3, log=print):
    """Upload `items` to `bucket` (a google.cloud.storage Bucket or anything with .blob(name) / .get_blob(name)).

    Returns (uploaded, skipped, failed, bytes_sent); `failed` is a list of (name, reason). A file counts as uploaded only when
    the MD5 GCS reports after the upload equals the local MD5.
    """
    uploaded, skipped, failed, sent = 0, 0, [], 0
    prefix = prefix.strip("/")
    for path, rel in items:
        name = f"{prefix}/{rel}" if prefix else rel
        size = path.stat().st_size
        want = md5_b64(path)
        have = bucket.get_blob(name)
        if have is not None and have.size == size and have.md5_hash == want:
            skipped += 1
            continue
        ok = False
        for attempt in range(1, retries + 1):
            try:
                blob = bucket.blob(name)
                blob.chunk_size = CHUNK
                blob.upload_from_filename(str(path), checksum="md5")
                blob.reload()
                if blob.md5_hash == want and blob.size == size:
                    ok = True
                    break
                failed_reason = f"md5/size mismatch after upload (local {want} {size}, remote {blob.md5_hash} {blob.size})"
            except Exception as exc:   # network errors, 5xx, expired credentials: retry, then report
                failed_reason = f"{type(exc).__name__}: {str(exc)[:160]}"
            log(f"  retry {attempt}/{retries} {name}: {failed_reason}")
            time.sleep(min(2 ** attempt, 30) if os.environ.get("GCS_PUSH_SLEEP", "1") == "1" else 0)
        if ok:
            uploaded += 1
            sent += size
        else:
            failed.append((name, failed_reason))
    return uploaded, skipped, failed, sent


def make_client(credentials=None, token_file=None, project=None):
    from google.cloud import storage   # imported late so the tests and --help work without the library
    credentials = credentials or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials:
        return storage.Client.from_service_account_json(credentials, project=project)
    if token_file:
        from google.oauth2.credentials import Credentials
        token = Path(token_file).read_text(encoding="utf-8").strip()
        return storage.Client(project=project or "unused", credentials=Credentials(token))
    return storage.Client(project=project)


def main(argv=None, client=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--prefix", default="", help="folder inside the bucket")
    ap.add_argument("--dir", default=None, help="upload every file under this folder (sub-folders kept)")
    ap.add_argument("--files", nargs="*", default=None, help="explicit files (uploaded under their base names)")
    ap.add_argument("--credentials", default=None, help="service-account JSON (default: $GOOGLE_APPLICATION_CREDENTIALS)")
    ap.add_argument("--token-file", default=None, help="file holding an OAuth access token")
    ap.add_argument("--project", default=None)
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args(argv)
    items = collect(args.dir, args.files)
    if not items:
        print("GCS_FAILED nothing to upload (check --dir / --files)")
        return 1
    try:
        client = client or make_client(args.credentials, args.token_file, args.project)
        bucket = client.bucket(args.bucket)
    except Exception as exc:
        print(f"GCS_FAILED cannot create the client or bucket handle: {type(exc).__name__}: {str(exc)[:200]}")
        return 1
    up, sk, failed, sent = push(bucket, args.prefix, items, retries=args.retries)
    for name, why in failed:
        print(f"FAILED {name}: {why}")
    if failed:
        print(f"GCS_FAILED uploaded={up} skipped={sk} failed={len(failed)} bytes={sent}")
        return 1
    print(f"GCS_OK uploaded={up} skipped={sk} bytes={sent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
