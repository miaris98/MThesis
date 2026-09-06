"""Unit tests for the experiment path resolver and the vast.ai -> local MLflow sync."""
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sync_experiments as sx
from src.config import paths


def write_experiment(store: Path, exp_id: str, name: str) -> Path:
    """Create an MLflow file-store experiment directory."""
    exp = store / exp_id
    exp.mkdir(parents=True, exist_ok=True)
    (exp / "meta.yaml").write_text(
        "artifact_location: file:///workspace/MThesis/mlruns/{0}\n"
        "creation_time: 1757000000000\n"
        "experiment_id: '{0}'\n"
        "last_update_time: 1757000000000\n"
        "lifecycle_stage: active\n"
        "name: {1}\n".format(exp_id, name),
        encoding="utf-8",
    )
    return exp


def write_run(exp: Path, run_id: str, with_params: bool = True) -> Path:
    """Create an MLflow file-store run directory under ``exp``."""
    run = exp / run_id
    for sub in ("artifacts", "metrics", "tags") + (("params",) if with_params else ()):
        (run / sub).mkdir(parents=True, exist_ok=True)
    (run / "meta.yaml").write_text(
        "artifact_uri: file:///workspace/MThesis/mlruns/{0}/{1}/artifacts\n"
        "end_time: 1757000900000\n"
        "experiment_id: '{0}'\n"
        "lifecycle_stage: active\n"
        "run_id: {1}\n"
        "run_name: run-{1}\n"
        "run_uuid: {1}\n"
        "start_time: 1757000100000\n"
        "status: 3\n"
        "tags: []\n"
        "user_id: root\n".format(exp.name, run_id),
        encoding="utf-8",
    )
    (run / "metrics" / "train_loss").write_text("1757000100000 0.5 0\n", encoding="utf-8")
    (run / "artifacts" / "summary.txt").write_text("artifact of " + run_id, encoding="utf-8")
    return run


class TestExperimentPaths(unittest.TestCase):
    """The resolver must pick the right root per machine and stay overridable."""

    def setUp(self):
        self._saved = os.environ.get(paths.ENV_VAR)

    def tearDown(self):
        if self._saved is None:
            os.environ.pop(paths.ENV_VAR, None)
        else:
            os.environ[paths.ENV_VAR] = self._saved

    def test_env_override_wins_and_shapes_the_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ[paths.ENV_VAR] = tmp
            self.assertEqual(paths.exp_root(), Path(tmp))
            self.assertEqual(paths.mlruns_dir(), Path(tmp) / "mlruns")
            self.assertEqual(paths.runs_dir(), Path(tmp) / "runs")
            self.assertEqual(paths.checkpoints_dir(), Path(tmp) / "checkpoints")
            self.assertEqual(paths.imports_dir(), Path(tmp) / "imports")

    def test_vastai_layout_is_unchanged_by_the_resolver(self):
        # The instance must keep writing where run_multi_carla_training.sh expects,
        # otherwise the tmux MLflow server serves an empty store.
        os.environ.pop(paths.ENV_VAR, None)
        if not paths._is_vastai():
            self.skipTest("not running on a /workspace instance")
        self.assertEqual(paths.runs_dir(), Path("/workspace/runs"))
        self.assertEqual(paths.checkpoints_dir(), Path("/workspace/checkpoints"))
        self.assertEqual(paths.mlruns_dir(), Path("/workspace/MThesis/mlruns"))

    def test_uri_is_a_valid_file_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            uri = paths.path_to_uri(Path(tmp))
            self.assertTrue(uri.startswith("file:///"), uri)
            self.assertNotIn("\\", uri)


class TestMetaYamlIO(unittest.TestCase):
    """The flat-YAML shim has to round-trip what MLflow's file store writes."""

    def test_reads_quoted_and_bare_scalars(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = Path(tmp) / "meta.yaml"
            meta.write_text("experiment_id: '3'\nname: WoR\nstatus: 3\ntags: []\n", encoding="utf-8")
            parsed = sx.read_meta(meta)
            self.assertEqual(parsed["experiment_id"], "3")
            self.assertEqual(parsed["name"], "WoR")
            self.assertEqual(parsed["status"], "3")

    def test_update_replaces_and_appends_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = Path(tmp) / "meta.yaml"
            meta.write_text("experiment_id: '1'\nname: WoR\n", encoding="utf-8")
            sx.update_meta(meta, {"experiment_id": "7", "artifact_uri": "file:///x/y"})
            parsed = sx.read_meta(meta)
            self.assertEqual(parsed["experiment_id"], "7")
            self.assertEqual(parsed["artifact_uri"], "file:///x/y")
            self.assertEqual(parsed["name"], "WoR")

    def test_single_quotes_in_values_are_escaped(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = Path(tmp) / "meta.yaml"
            meta.write_text("name: x\n", encoding="utf-8")
            sx.update_meta(meta, {"name": "it's fine"})
            self.assertEqual(sx.read_meta(meta)["name"], "it''s fine")
            raw = meta.read_text(encoding="utf-8")
            self.assertIn("'it''s fine'", raw)


class TestMergeStore(unittest.TestCase):
    """Merging remote runs into the local archive must not lose or collide runs."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.src = self.tmp / "snap" / "MThesis" / "mlruns"
        self.dst = self.tmp / "archive" / "mlruns"

        write_experiment(self.src, "0", "Default")
        remote_wor = write_experiment(self.src, "1", "WoR_Offline_Training")
        remote_ppo = write_experiment(self.src, "2", "CARLA_PPO_RL")
        write_run(remote_wor, "a" * 32)
        write_run(remote_wor, "b" * 32)
        # A run that logged no params arrives without that folder.
        write_run(remote_ppo, "c" * 32, with_params=False)

        # The local archive already holds a *different* experiment at id 1.
        write_experiment(self.dst, "0", "Default")
        write_experiment(self.dst, "1", "Local_Sanity")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_colliding_experiment_ids_are_remapped_by_name(self):
        merged, skipped = sx.merge_store(self.src, self.dst, "inst-1", verbose=False)
        self.assertEqual((merged, skipped), (3, 0))

        by_name = sx.experiments_by_name(self.dst)
        # Local_Sanity keeps id 1; the remote experiments get fresh ids.
        self.assertEqual(by_name["Local_Sanity"].name, "1")
        self.assertEqual(by_name["Default"].name, "0")
        self.assertNotIn(by_name["WoR_Offline_Training"].name, ("0", "1"))
        self.assertNotEqual(by_name["WoR_Offline_Training"].name,
                            by_name["CARLA_PPO_RL"].name)

    def test_free_source_id_is_preserved(self):
        # MLflow 3 hands out large random experiment ids, which do not collide.
        # Those should carry over unchanged so a run keeps its remote coordinates.
        random_id = "440167414335841706"
        write_run(write_experiment(self.src, random_id, "Qwen_WoR"), "e" * 32)
        sx.merge_store(self.src, self.dst, "inst-1", verbose=False)
        self.assertEqual(sx.experiments_by_name(self.dst)["Qwen_WoR"].name, random_id)

    def test_artifact_uris_point_at_the_new_location(self):
        sx.merge_store(self.src, self.dst, "inst-1", verbose=False)
        wor = sx.experiments_by_name(self.dst)["WoR_Offline_Training"]
        run = wor / ("a" * 32)
        meta = sx.read_meta(run / "meta.yaml")
        self.assertEqual(meta["artifact_uri"], paths.path_to_uri(run / "artifacts"))
        self.assertEqual(meta["experiment_id"], wor.name)
        self.assertNotIn("workspace", meta["artifact_uri"])
        self.assertTrue((run / "artifacts" / "summary.txt").is_file())

    def test_provenance_tag_records_the_source_instance(self):
        sx.merge_store(self.src, self.dst, "ssh5-12345", verbose=False)
        wor = sx.experiments_by_name(self.dst)["WoR_Offline_Training"]
        tag = wor / ("a" * 32) / "tags" / "source_instance"
        self.assertEqual(tag.read_text(encoding="utf-8"), "ssh5-12345")

    def test_missing_run_subdirs_are_recreated(self):
        # MLflow's FileStore hides any run lacking metrics/params/artifacts.
        sx.merge_store(self.src, self.dst, "inst-1", verbose=False)
        ppo = sx.experiments_by_name(self.dst)["CARLA_PPO_RL"]
        run = ppo / ("c" * 32)
        for required in ("metrics", "params", "artifacts"):
            self.assertTrue((run / required).is_dir(), required + " missing")

    def test_reserved_experiment_folders_are_copied_not_treated_as_runs(self):
        # MLflow keeps experiment tags, datasets, traces and registered models
        # beside the run directories; treating one as a run would corrupt it.
        wor_src = self.src / "1"
        (wor_src / "tags" / "mlflow.note").parent.mkdir(parents=True, exist_ok=True)
        (wor_src / "tags" / "mlflow.note").write_text("experiment note", encoding="utf-8")
        (wor_src / "traces").mkdir(exist_ok=True)

        merged, _ = sx.merge_store(self.src, self.dst, "inst-1", verbose=False)
        self.assertEqual(merged, 3, "reserved folders must not be counted as runs")

        wor_dst = sx.experiments_by_name(self.dst)["WoR_Offline_Training"]
        self.assertEqual((wor_dst / "tags" / "mlflow.note").read_text(encoding="utf-8"),
                         "experiment note")
        self.assertTrue((wor_dst / "traces").is_dir())
        self.assertFalse((wor_dst / "tags" / "meta.yaml").exists(),
                         "a reserved folder must not be given run metadata")

    def test_second_merge_is_a_no_op(self):
        sx.merge_store(self.src, self.dst, "inst-1", verbose=False)
        merged, skipped = sx.merge_store(self.src, self.dst, "inst-1", verbose=False)
        self.assertEqual((merged, skipped), (0, 3))

    def test_overwrite_replaces_existing_runs(self):
        sx.merge_store(self.src, self.dst, "inst-1", verbose=False)
        merged, skipped = sx.merge_store(self.src, self.dst, "inst-2", overwrite=True, verbose=False)
        self.assertEqual((merged, skipped), (3, 0))
        wor = sx.experiments_by_name(self.dst)["WoR_Offline_Training"]
        tag = wor / ("a" * 32) / "tags" / "source_instance"
        self.assertEqual(tag.read_text(encoding="utf-8"), "inst-2")

    def test_relink_repairs_paths_after_a_move(self):
        sx.merge_store(self.src, self.dst, "inst-1", verbose=False)
        moved = self.tmp / "moved" / "mlruns"
        moved.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(self.dst), str(moved))

        fixed = sx.relink(moved)
        self.assertGreater(fixed, 0)
        wor = sx.experiments_by_name(moved)["WoR_Offline_Training"]
        run = wor / ("a" * 32)
        self.assertEqual(sx.read_meta(run / "meta.yaml")["artifact_uri"],
                         paths.path_to_uri(run / "artifacts"))

    def test_reserved_directories_are_not_treated_as_experiments(self):
        (self.src / ".trash").mkdir(exist_ok=True)
        (self.src / "models").mkdir(exist_ok=True)
        (self.src / "models" / "meta.yaml").write_text("name: models\n", encoding="utf-8")
        names = set(sx.experiments_by_name(self.src))
        self.assertNotIn("models", names)


class TestRemoteTransfer(unittest.TestCase):
    """The remote tar snippet and local extraction, exercised without a real instance."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.remote = self.tmp / "workspace"
        (self.remote / "runs").mkdir(parents=True)
        (self.remote / "runs" / "training_telemetry.csv").write_text("step,loss\n1,0.5\n",
                                                                     encoding="utf-8")
        write_run(write_experiment(self.remote / "MThesis" / "mlruns", "1", "WoR_Offline_Training"),
                  "a" * 32)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run_remote(self, script: str, out: Path = None):
        """Execute the generated remote snippet through a POSIX shell."""
        sh = shutil.which("bash") or shutil.which("sh")
        if sh is None:
            self.skipTest("no POSIX shell available to emulate the remote side")
        sink = open(out, "wb") if out else subprocess.DEVNULL
        try:
            return subprocess.run([sh, "-c", script], stdout=sink).returncode
        finally:
            if out:
                sink.close()

    def test_missing_paths_are_skipped_not_fatal(self):
        # checkpoints/ and the videos do not exist yet - the usual mid-run case.
        script = sx.build_remote_tar_cmd(self.remote.as_posix(), sx.resolve_includes(["all"]))
        archive = self.tmp / "out.tar.gz"
        self.assertEqual(self._run_remote(script, archive), 0)

        dest = self.tmp / "snap"
        sx.extract_snapshot(archive, dest)
        self.assertTrue((dest / "MThesis" / "mlruns" / "1" / "meta.yaml").is_file())
        self.assertTrue((dest / "runs" / "training_telemetry.csv").is_file())
        self.assertEqual(sx.find_mlruns(dest), dest / "MThesis" / "mlruns")

    def test_sentinel_exit_codes(self):
        nothing = sx.build_remote_tar_cmd(self.remote.as_posix(), ["no/such/path"])
        self.assertEqual(self._run_remote(nothing), 8, "empty selection should exit 8")
        bad_root = sx.build_remote_tar_cmd("/definitely/not/a/real/root", ["x"])
        self.assertEqual(self._run_remote(bad_root), 9, "missing root should exit 9")

    def test_extraction_rejects_paths_outside_the_destination(self):
        # A malicious or malformed archive must not write above dest. Only meaningful
        # where tarfile's data filter exists (3.12+); older runtimes are skipped.
        if sys.version_info < (3, 12):
            self.skipTest("tarfile data filter requires Python 3.12+")
        archive = self.tmp / "evil.tar.gz"
        payload = self.tmp / "payload.txt"
        payload.write_text("nope", encoding="utf-8")
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(payload, arcname="../escaped.txt")
        with self.assertRaises(Exception):
            sx.extract_snapshot(archive, self.tmp / "dest")


class TestSshParsing(unittest.TestCase):
    """Accept the strings vast.ai actually hands out."""

    def test_parses_full_ssh_command(self):
        self.assertEqual(sx.parse_ssh_cmd("ssh -p 38472 root@ssh5.vast.ai"),
                         ("38472", "root@ssh5.vast.ai"))

    def test_defaults_port_when_absent(self):
        port, host = sx.parse_ssh_cmd("ssh root@ssh5.vast.ai")
        self.assertEqual((port, host), ("22", "root@ssh5.vast.ai"))

    def test_snapshot_tag_is_filesystem_safe(self):
        tag = sx.default_tag("root@ssh5.vast.ai", "38472")
        self.assertTrue(tag.startswith("ssh5-38472_"))
        for bad in '<>:"/\\|?*@.':
            self.assertNotIn(bad, tag)

    def test_unknown_include_group_is_rejected(self):
        with self.assertRaises(SystemExit):
            sx.resolve_includes(["definitely-not-a-group"])

    def test_all_expands_to_every_group(self):
        expanded = sx.resolve_includes(["all"])
        for group_paths in sx.INCLUDE_GROUPS.values():
            for p in group_paths:
                self.assertIn(p, expanded)


if __name__ == "__main__":
    unittest.main()
