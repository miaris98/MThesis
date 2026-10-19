"""scripts/sync/gcs_push.py: upload logic with a fake bucket (no network, no google library needed)."""
import base64
import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "sync"))
os.environ["GCS_PUSH_SLEEP"] = "0"

import gcs_push  # noqa: E402


def _md5(data: bytes) -> str:
    return base64.b64encode(hashlib.md5(data).digest()).decode()


class FakeBlob:
    def __init__(self, store, name, corrupt=False, fail_times=0):
        self.store, self.name, self.corrupt, self.fail_times = store, name, corrupt, fail_times
        self.chunk_size = None
        d = store.get(name)
        self.size, self.md5_hash = (len(d), _md5(d)) if d is not None else (None, None)

    def upload_from_filename(self, path, checksum=None):
        assert checksum == "md5"
        if self.fail_times > 0:
            self.fail_times -= 1
            raise ConnectionError("simulated drop")
        data = Path(path).read_bytes()
        self.store[self.name] = data[:-1] if self.corrupt else data   # a corrupt upload stores fewer bytes

    def reload(self):
        d = self.store[self.name]
        self.size, self.md5_hash = len(d), _md5(d)


class FakeBucket:
    def __init__(self, preset=None, corrupt=(), flaky=None):
        self.store = dict(preset or {})
        self.corrupt, self.flaky = set(corrupt), dict(flaky or {})
        self.uploads = 0

    def get_blob(self, name):
        return FakeBlob(self.store, name) if name in self.store else None

    def blob(self, name):
        self.uploads += 1
        b = FakeBlob(self.store, name, corrupt=name in self.corrupt, fail_times=self.flaky.get(name, 0))
        if b.fail_times:
            self.flaky[name] = 0   # the failure is consumed by this single blob handle
        return b


def _tree(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.json").write_bytes(b"aaa")
    (tmp_path / "sub" / "b.pt").write_bytes(b"bbbb")
    (tmp_path / "c.tmp").write_bytes(b"half")
    (tmp_path / "d.tmp.npz").write_bytes(b"half")
    return tmp_path


def test_collect_skips_in_progress_files_and_keeps_subfolders(tmp_path):
    names = sorted(rel for _, rel in gcs_push.collect(_tree(tmp_path), None))
    assert names == ["a.json", "sub/b.pt"]


def test_upload_then_second_run_skips_everything(tmp_path):
    items = gcs_push.collect(_tree(tmp_path), None)
    bucket = FakeBucket()
    up, sk, failed, sent = gcs_push.push(bucket, "p/x", items, log=lambda *_: None)
    assert (up, sk, failed, sent) == (2, 0, [], 7)
    assert bucket.store["p/x/a.json"] == b"aaa" and bucket.store["p/x/sub/b.pt"] == b"bbbb"
    up, sk, failed, sent = gcs_push.push(bucket, "p/x", items, log=lambda *_: None)
    assert (up, sk, failed, sent) == (0, 2, [], 0)


def test_changed_file_is_uploaded_again(tmp_path):
    items = gcs_push.collect(_tree(tmp_path), None)
    bucket = FakeBucket(preset={"a.json": b"old", "sub/b.pt": b"bbbb"})
    up, sk, failed, _ = gcs_push.push(bucket, "", items, log=lambda *_: None)
    assert (up, sk, failed) == (1, 1, [])
    assert bucket.store["a.json"] == b"aaa"


def test_md5_mismatch_after_upload_is_reported_as_failure(tmp_path):
    items = gcs_push.collect(_tree(tmp_path), None)
    bucket = FakeBucket(corrupt={"a.json"})
    up, sk, failed, _ = gcs_push.push(bucket, "", items, retries=2, log=lambda *_: None)
    assert up == 1 and len(failed) == 1 and failed[0][0] == "a.json" and "mismatch" in failed[0][1]


def test_transient_error_is_retried(tmp_path):
    items = gcs_push.collect(_tree(tmp_path), None)
    bucket = FakeBucket(flaky={"a.json": 1})
    up, sk, failed, _ = gcs_push.push(bucket, "", items, retries=3, log=lambda *_: None)
    assert (up, failed) == (2, [])


def test_main_exit_codes(tmp_path, capsys):
    root = _tree(tmp_path)

    class FakeClient:
        def __init__(self, bucket):
            self._b = bucket

        def bucket(self, name):
            return self._b

    good = gcs_push.main(["--bucket", "b", "--dir", str(root)], client=FakeClient(FakeBucket()))
    assert good == 0 and "GCS_OK uploaded=2 skipped=0 bytes=7" in capsys.readouterr().out
    bad = gcs_push.main(["--bucket", "b", "--dir", str(root), "--retries", "1"], client=FakeClient(FakeBucket(corrupt={"a.json"})))
    assert bad == 1 and "GCS_FAILED" in capsys.readouterr().out
    empty = gcs_push.main(["--bucket", "b", "--dir", str(root / "sub" / "nothing")], client=FakeClient(FakeBucket()))
    assert empty == 1
