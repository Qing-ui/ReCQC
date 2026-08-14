import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "apps" / "recqc-2.0"
sys.path.insert(0, str(APP))

from CarbonScoreProcess import GreedyMatcher  # noqa: E402
from HSQCScoreProcess import UserPointMatcher  # noqa: E402


def reference_distances(db_points, user_points):
    distances = []
    for i, (db_c, db_h) in enumerate(db_points):
        for j, (user_c, user_h) in enumerate(user_points):
            delta_h = (db_h - user_h) * 10
            delta_c = db_c - user_c
            distance = (delta_c**2 + delta_h**2) ** 0.5
            distances.append((i, j, round(distance, 2)))
    return sorted(distances, key=lambda item: item[2])


class Recqc20EquivalenceTests(unittest.TestCase):
    def test_vectorized_hsqc_distances_match_scalar_reference(self):
        db_points = [(100.0, 1.0), (100.0, 1.2), (102.345, 2.125)]
        user_points = [(100.0, 1.1), (101.25, 2.0), (99.5, 0.95)]
        matcher = object.__new__(UserPointMatcher)

        self.assertEqual(
            reference_distances(db_points, user_points),
            matcher._calculate_distances(db_points, user_points),
        )

    def test_carbon_matching_uses_one_writer_and_keeps_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first.db"
            second = Path(temp_dir) / "second.db"

            first_rows = self._run_synthetic_match(first, workers=4)
            second_rows = self._run_synthetic_match(second, workers=2)

        expected = [
            (mol_id, f"M{mol_id}", 10.1, 0, 10.0, 1, "[]", "[]") for mol_id in range(1, 4)
        ] + [(mol_id, f"M{mol_id}", 20.2, 1, 20.0, 1, "[]", "[]") for mol_id in range(1, 4)]
        expected.sort(key=lambda row: (row[0], row[3]))

        self.assertEqual(expected, first_rows)
        self.assertEqual(expected, second_rows)

    @staticmethod
    def _run_synthetic_match(db_path, workers):
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                CREATE TABLE molecules (
                    mol_id INTEGER PRIMARY KEY,
                    name TEXT,
                    carbon_count INTEGER,
                    fw REAL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE carbons (
                    carbon_id INTEGER PRIMARY KEY,
                    mol_id INTEGER,
                    c_index INTEGER,
                    h_count INTEGER,
                    c_shift REAL,
                    h_shifts TEXT,
                    connected_carbon TEXT
                )
                """
            )
            for mol_id in range(1, 4):
                conn.execute(
                    "INSERT INTO molecules VALUES (?, ?, 2, 100.0)",
                    (mol_id, f"M{mol_id}"),
                )
                conn.executemany(
                    """
                    INSERT INTO carbons
                    (mol_id, c_index, h_count, c_shift, h_shifts, connected_carbon)
                    VALUES (?, ?, 0, ?, '[]', '[]')
                    """,
                    ((mol_id, 0, 10.0), (mol_id, 1, 20.0)),
                )
        conn.close()

        matcher = GreedyMatcher(
            str(db_path),
            [10.1, 20.2],
            c_range=(1, 2),
            fw_range=(0.0, 200.0),
            mode="untyped",
            C_merge="N",
        )
        matcher.run(max_workers=workers)

        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                """
                SELECT mol_id, mol_name, user_shift, db_c_index, db_shift,
                       type_matched, connected_h_shifts, connected_carbons
                FROM greedy_matches
                ORDER BY mol_id, db_c_index
                """
            ).fetchall()
        conn.close()
        return rows


if __name__ == "__main__":
    unittest.main()
