import importlib.util
import unittest
from itertools import product
from pathlib import Path


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

    def test_shared_desktop_and_scoring_files_are_identical(self):
        for name in (
            "gui.py",
            "PROCESSSDFFILES.py",
            "SDFFileSelector.py",
            "CarbonScorerResult.py",
            "HSQCScorerResult.py",
            "CombineScorerResult.py",
            "img.png",
        ):
            with self.subTest(name=name):
                self.assertEqual((ORIGINAL / name).read_bytes(), (SPEED / name).read_bytes())

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
