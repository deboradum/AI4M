import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_split import choose_validation  # noqa: E402
from slice_segthor import get_splits  # noqa: E402


def fake_rows(n: int = 40) -> list[dict]:
    rows = []
    for i in range(1, n + 1):
        rows.append({"id": f"Patient_{i:02d}",
                     "contrast": i % 3 == 0,            # 13 of 40
                     "spacing_outlier": i in (11, 15, 20, 35),
                     "volume_outlier": i in (18, 28, 34)})
    return rows


class TestChooseValidation(unittest.TestCase):
    def test_size_and_strata(self):
        rows = fake_rows()
        val, seed = choose_validation(rows, n_val=8, seed=0)
        self.assertEqual(len(val), 8)
        self.assertEqual(len(set(val)), 8)
        by = {r["id"]: r for r in rows}
        self.assertEqual(sum(by[v]["contrast"] for v in val), round(13 * 8 / 40))   # cohort share: 3
        self.assertGreaterEqual(sum(by[v]["spacing_outlier"] for v in val), 1)
        self.assertLessEqual(sum(by[v]["spacing_outlier"] for v in val), 2)
        self.assertEqual(sum(by[v]["volume_outlier"] for v in val), 1)

    def test_deterministic(self):
        rows = fake_rows()
        self.assertEqual(choose_validation(rows, 8, seed=0), choose_validation(rows, 8, seed=0))

    def test_impossible_constraints_raise(self):
        # One spacing outlier and one (different) volume outlier must both be in a
        # validation set of size one: no draw can satisfy that.
        rows = [{"id": f"Patient_{i:02d}", "contrast": False,
                 "spacing_outlier": i == 1, "volume_outlier": i == 2} for i in range(1, 11)]
        with self.assertRaises(RuntimeError):
            choose_validation(rows, n_val=1, max_tries=50)


class TestSlicerSplitFile(unittest.TestCase):
    def test_split_file_overrides_random_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp)
            ids = [f"Patient_{i:02d}" for i in range(1, 11)]
            for pid in ids:
                (src / "train" / pid).mkdir(parents=True)
            (src / "test").mkdir()
            split = src / "split.json"
            split.write_text(json.dumps({"validation": ["Patient_02", "Patient_07"]}))

            train, val, test = get_splits(src, retains=5, fold=0, split_file=split)
            self.assertEqual(val, ["Patient_02", "Patient_07"])
            self.assertEqual(sorted(train), sorted(set(ids) - set(val)))
            self.assertEqual(test, [])

    def test_unknown_patient_in_split_file_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp)
            (src / "train" / "Patient_01").mkdir(parents=True)
            (src / "test").mkdir()
            split = src / "split.json"
            split.write_text(json.dumps({"validation": ["Patient_42"]}))
            with self.assertRaises(AssertionError):
                get_splits(src, retains=1, fold=0, split_file=split)


if __name__ == "__main__":
    unittest.main()
