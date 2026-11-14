"""Public arithmetic reconstruction and malformed-evidence rejection checks."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import shutil
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("h100_public_analysis", ROOT / "tools/analyze_h100_timing.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class PublicH100AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [json.loads(line) for line in
                    (analysis.DEFAULT_DATA / "measurements.jsonl").read_text().splitlines()]

    def test_actual_public_observations_reproduce_historical_statistics(self):
        result = analysis.analyze()
        self.assertEqual(result["status"], "historical_numerical_statistics_reproduced")
        self.assertEqual(result["observations"], 400)
        self.assertEqual(result["comparison"]["independent_scenes"], 50)
        self.assertEqual(result["recorded_fields_checked"], 48)
        self.assertFalse(result["comparison"]["quality_preserving"])

    def test_linear_quantile_has_known_values(self):
        self.assertEqual(analysis.quantile([0., 10., 20., 30.], .5), 15.)
        self.assertAlmostEqual(analysis.quantile([0., 10., 20., 30.], .95), 28.5)
        self.assertEqual(analysis.quantile([7.], .95), 7.)

    def test_incomplete_dataset_is_rejected(self):
        with self.assertRaises(ValueError):
            analysis.checked_rows(self.rows[:-1])

    def test_duplicate_scene_replaces_independent_scene_is_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["scene_index"] = rows[1]["scene_index"]
        with self.assertRaises(ValueError):
            analysis.checked_rows(rows)

    def test_unpaired_seed_is_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[50]["seed"] += 1
        with self.assertRaises(ValueError):
            analysis.checked_rows(rows)

    def test_missing_work_or_failed_requests_are_rejected(self):
        for field, value in (("denoiser_evaluations", 49), ("status", "error"),
                             ("failure_count", 1), ("fallback_count", 1),
                             ("oom_attempt_count", 1), ("repeat", 1)):
            with self.subTest(field=field):
                rows = copy.deepcopy(self.rows)
                rows[0][field] = value
                with self.assertRaises(ValueError):
                    analysis.checked_rows(rows)

    def test_invalid_numeric_observations_are_rejected(self):
        for value in (0, -1, float("nan"), float("inf"), True, "57"):
            with self.subTest(value=value):
                rows = copy.deepcopy(self.rows)
                rows[0]["latency_s"] = value
                with self.assertRaises(ValueError):
                    analysis.checked_rows(rows)

    def test_request_order_and_extra_fields_are_rejected(self):
        for mutate in (lambda r: r.update(request_index=1),
                       lambda r: r.update(private_host="example")):
            rows = copy.deepcopy(self.rows)
            mutate(rows[0])
            with self.assertRaises(ValueError):
                analysis.checked_rows(rows)

    def test_incorrect_recorded_statistics_are_rejected(self):
        with self.assertRaises(ValueError):
            analysis.compare_recorded({"latency": 43.}, {"latency": 42.})

    def test_bootstrap_wrong_coverage_is_rejected(self):
        with self.assertRaises(ValueError):
            analysis.checked_draws("0,1,2\n")
        with self.assertRaises(ValueError):
            analysis.checked_draws((",".join(["50"] * 50) + "\n") * 4000)

    def test_changed_file_bytes_are_rejected_before_analysis(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder) / "audit"
            shutil.copytree(analysis.DEFAULT_DATA, directory)
            with (directory / "measurements.jsonl").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "checksum differs"):
                analysis.analyze(directory)

    def test_missing_manifest_binding_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder) / "audit"
            shutil.copytree(analysis.DEFAULT_DATA, directory)
            path = directory / "manifest.json"
            manifest = json.loads(path.read_text())
            del manifest["files"]["measurements.jsonl"]
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "bind every"):
                analysis.analyze(directory)


if __name__ == "__main__":
    unittest.main()
