import importlib.util
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from itertools import product
from pathlib import Path
from unittest.mock import patch

from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "apps" / "recqc-2.0"
SPEED = ROOT / "ReCQC2.0-speed"


def load(directory, module):
    spec = importlib.util.spec_from_file_location(directory.name + "_" + module, directory / module)
    instance = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(instance)
    return instance


class Recqc20SpeedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.carbon = [
            load(directory, "CarbonScoreProcess.py").GreedyMatcher
            for directory in (ORIGINAL, SPEED)
        ]
        cls.hsqc = [
            load(directory, "HSQCScoreProcess.py").UserPointMatcher
            for directory in (ORIGINAL, SPEED)
        ]

    def test_shared_input_assets_are_identical(self):
        for name in (
            "SDFFileSelector.py",
            "img.png",
        ):
            with self.subTest(name=name):
                self.assertEqual((ORIGINAL / name).read_bytes(), (SPEED / name).read_bytes())

    def test_prepared_library_preserves_database_and_invalidates_on_same_size_edit(self):
        original = load(ORIGINAL, "PROCESSSDFFILES.py").SDFBatchProcessor
        speed = load(SPEED, "PROCESSSDFFILES.py").SDFBatchProcessor
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            folder = Path(temp_dir)
            source = folder / "library.sdf"
            live = folder / "live.sqlite"
            baseline = folder / "original.sqlite"

            mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
            mol.SetProp("ID", "first")
            mol.SetProp("FW", "30.07")
            mol.SetProp("Predicted 13C shifts", "1C[1] 10.0\n2C[2] 20.0")
            mol.SetProp("HydrogenShifts", "1a[3] 1.0")
            with Chem.SDWriter(str(source)) as writer:
                writer.write(mol)
                mol.ClearProp("FW")
                writer.write(mol)

            def dump(path):
                with closing(sqlite3.connect(path)) as conn:
                    return list(conn.iterdump())

            paths = [str(source)]
            expected_report = original.process_files_atomically(paths, str(baseline))
            self.assertEqual((expected_report["accepted"], expected_report["rejected"]), (1, 1))
            report = speed.process_files_atomically(paths, str(live))
            self.assertEqual(report, expected_report)
            self.assertEqual(dump(live), dump(baseline))
            with closing(sqlite3.connect(live)) as conn:
                conn.execute("CREATE TABLE prior_result (score REAL)")
                conn.commit()
            report["errors"].clear()
            with patch.object(speed, "process_files", side_effect=AssertionError("expected reuse")):
                self.assertEqual(speed.process_files_atomically(paths, str(live)), expected_report)
            self.assertEqual(dump(live), dump(baseline))

            old_stat = source.stat()
            payload = source.read_bytes()
            updated = payload.replace(b"10.0", b"15.0")
            self.assertNotEqual(payload, updated)
            source.write_bytes(updated)
            os.utime(source, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
            expected_report = original.process_files_atomically(paths, str(baseline))
            self.assertEqual(speed.process_files_atomically(paths, str(live)), expected_report)
            self.assertEqual(dump(live), dump(baseline))

            speed._prepared[1].write_bytes(b"invalid cache")
            self.assertEqual(speed.process_files_atomically(paths, str(live)), expected_report)
            next_source = folder / "next-library.sdf"
            next_source.write_bytes(updated.replace(b"first", b"second"))
            next_paths = [str(next_source)]
            new_report = original.process_files_atomically(next_paths, str(baseline))
            self.assertEqual(speed.process_files_atomically(next_paths, str(live)), new_report)
            expected_db = dump(baseline)
            self.assertEqual(dump(live), expected_db)
            with closing(sqlite3.connect(live)) as conn:
                self.assertEqual(
                    conn.execute("SELECT name,source_file FROM molecules").fetchall(),
                    [("second", str(next_source))],
                )
            source.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "No valid molecules"):
                speed.process_files_atomically(paths, str(live))
            self.assertEqual(dump(live), expected_db)

    def test_carbon_pairs_keep_original_order_and_type_fallback(self):
        carbons = {
            i: {
                "c_index": i,
                "c_shift": shift,
                "h_count": h_count,
                "h_shifts": "[]",
                "connected_c": "[]",
            }
            for i, (shift, h_count) in enumerate(
                [(10.0, 3), (10.0, 0), (10.0, 3), (9.5, 1), (15.0, 2), (20.0, 1)]
            )
        }
        typed = {0: [9.5, 10.5], 1: [], 2: [16.0, 16.0], 3: [10.5, 9.5]}
        untyped = [10.5, 9.5, 10.5, 16.0]
        for merge, mode in product(("N", "Y"), ("typed", "untyped")):
            with self.subTest(merge=merge, mode=mode):
                rows = []
                for matcher_type in self.carbon:
                    matcher = object.__new__(matcher_type)
                    matcher.mode, matcher.C_merge = mode, merge
                    matcher.carbon_cache = {1: {"name": "fixture", "carbons": carbons}}
                    matcher.user_data = typed if mode == "typed" else untyped
                    rows.append(matcher._process_molecule(1))
                self.assertTrue(rows[0])
                self.assertEqual(rows[0], rows[1])

    def test_hsqc_pairs_keep_ties_groups_and_placeholders(self):
        small = [(1, 1, (10.0, 1.0)), (2, 3, (10.0, 1.0)), (3, 2, (10.0, 1.2))]
        points = [(10.0, 1.1), (11.125, 1.0)]
        large = [(i, 1, (10.0 + i / 8, 1.0)) for i in range(80)]
        large_query = [(10.0 + i / 8 + 0.015, 1.1) for i in range(70)]
        unusual = [(1, 1, (float("inf"), 1.0)), (2, 2, (float("nan"), 2.0)), *small]
        for merge, (db, query) in product(
            ("N", "Y"),
            [(small, points), (small, []), ([], points), (large, large_query), (unusual, points)],
        ):
            with self.subTest(merge=merge, reference=len(db), query=len(query)):
                rows = []
                for matcher_type in self.hsqc:
                    matcher = object.__new__(matcher_type)
                    matcher.CH_merge = merge
                    rows.append(matcher._process_single_molecule(1, db, query))
                # repr makes NaN comparisons meaningful and preserves row order.
                self.assertEqual(repr(rows[0]), repr(rows[1]))


if __name__ == "__main__":
    unittest.main()
