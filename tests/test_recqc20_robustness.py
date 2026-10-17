import importlib.util
import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
VERSIONS = {
    "recqc20": ROOT / "apps" / "recqc-2.0",
    "hungarian": ROOT / "apps" / "recqc-hungarian",
    "speed": ROOT / "ReCQC2.0-speed",
}
RECQC1 = ROOT / "apps" / "recqc-1.0"


def load_module(version, filename):
    path = (RECQC1 if version == "recqc1" else VERSIONS[version]) / filename
    module_name = f"test_{version}_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SymmetryFilterTests(unittest.TestCase):
    def test_merge_mode_filters_on_distinct_signals_in_every_2x_version(self):
        for version in VERSIONS:
            with self.subTest(version=version), tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
                db_path = Path(temp_dir) / "symmetry.db"
                self._create_symmetry_db(db_path)
                matcher_class = load_module(version, "CarbonScoreProcess.py").GreedyMatcher
                matcher = matcher_class(
                    str(db_path),
                    [10.1, 20.1, 30.1, 40.1],
                    c_range=(1, 10),
                    fw_range=(0.0, 500.0),
                    mode="untyped",
                    C_merge="Y",
                )
                matcher.run(max_workers=1)

                conn = sqlite3.connect(db_path)
                try:
                    matched_molecules = conn.execute(
                        "SELECT DISTINCT mol_id FROM greedy_matches ORDER BY mol_id"
                    ).fetchall()
                    match_count = conn.execute(
                        "SELECT COUNT(*) FROM greedy_matches WHERE mol_id=1"
                    ).fetchone()[0]
                finally:
                    conn.close()

                self.assertEqual([(1,)], matched_molecules)
                self.assertEqual(6, match_count)

    @staticmethod
    def _create_symmetry_db(db_path):
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                "CREATE TABLE molecules (mol_id INTEGER PRIMARY KEY, name TEXT, carbon_count INTEGER, fw REAL)"
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
            conn.execute("INSERT INTO molecules VALUES (1, 'symmetric', 6, 100.0)")
            conn.execute("INSERT INTO molecules VALUES (2, 'too_many_signals', 6, 100.0)")
            conn.executemany(
                """
                INSERT INTO carbons
                (mol_id, c_index, h_count, c_shift, h_shifts, connected_carbon)
                VALUES (?, ?, 0, ?, '[]', '[]')
                """,
                [(1, index, shift) for index, shift in enumerate([10, 10, 20, 20, 30, 40])]
                + [(2, index, shift) for index, shift in enumerate([10, 20, 30, 40, 50, 50])],
            )
            conn.commit()
        finally:
            conn.close()


class SdfIsolationTests(unittest.TestCase):
    def test_bad_record_is_skipped_without_stopping_later_records(self):
        for version in VERSIONS:
            with self.subTest(version=version), tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
                temp_path = Path(temp_dir)
                sdf_path = temp_path / "mixed.sdf"
                self._write_mixed_sdf(sdf_path)
                db_path = temp_path / "import.db"
                processor_class = load_module(version, "PROCESSSDFFILES.py").SDFBatchProcessor
                processor = processor_class(str(db_path))
                report = processor.process_files([str(sdf_path)])

                conn = sqlite3.connect(db_path)
                try:
                    names = conn.execute("SELECT name FROM molecules ORDER BY mol_id").fetchall()
                    carbon_rows = conn.execute("SELECT COUNT(*) FROM carbons").fetchone()[0]
                finally:
                    conn.close()

                self.assertEqual(3, report["total"])
                self.assertEqual(2, report["accepted"])
                self.assertEqual(1, report["rejected"])
                self.assertEqual([("valid-1",), ("valid-2",)], names)
                self.assertEqual(4, carbon_rows)

    def test_all_bad_import_preserves_the_previous_database(self):
        for version in VERSIONS:
            with self.subTest(version=version), tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
                temp_path = Path(temp_dir)
                sdf_path = temp_path / "bad.sdf"
                writer = Chem.SDWriter(str(sdf_path))
                try:
                    mol = Chem.MolFromSmiles("CC")
                    mol.SetProp("ID", "missing-fw")
                    mol.SetProp("Predicted 13C shifts", "1C[1] 10.0\n2C[2] 20.0")
                    writer.write(mol)
                finally:
                    writer.close()

                db_path = temp_path / "live.db"
                conn = sqlite3.connect(db_path)
                try:
                    conn.execute("CREATE TABLE sentinel (value TEXT)")
                    conn.execute("INSERT INTO sentinel VALUES ('keep-me')")
                    conn.commit()
                finally:
                    conn.close()

                processor_class = load_module(version, "PROCESSSDFFILES.py").SDFBatchProcessor
                with self.assertRaisesRegex(ValueError, "No valid molecules"):
                    processor_class.process_files_atomically([str(sdf_path)], str(db_path))

                conn = sqlite3.connect(db_path)
                try:
                    value = conn.execute("SELECT value FROM sentinel").fetchone()[0]
                finally:
                    conn.close()
                self.assertEqual("keep-me", value)
                self.assertEqual([], list(temp_path.glob("*.importing")))

    @staticmethod
    def _write_mixed_sdf(path):
        writer = Chem.SDWriter(str(path))
        try:
            for name, include_fw in (("valid-1", True), ("bad", False), ("valid-2", True)):
                mol = Chem.MolFromSmiles("CC")
                mol.SetProp("ID", name)
                if include_fw:
                    mol.SetProp("FW", "30.0")
                mol.SetProp("Predicted 13C shifts", "1C[1] 10.0\n2C[2] 20.0")
                writer.write(mol)
        finally:
            writer.close()


class Recqc1BadMoleculeTests(unittest.TestCase):
    def test_malformed_shift_row_is_skipped_instead_of_exiting(self):
        sys.path.insert(0, str(RECQC1))
        try:
            sdf_module = load_module("recqc1", "SDF.py")
        finally:
            sys.path.pop(0)

        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            sdf_path = Path(temp_dir) / "recqc1-mixed.sdf"
            writer = Chem.SDWriter(str(sdf_path))
            try:
                for name, shifts in (
                    ("valid", "1\t10.0\n2\t20.0"),
                    ("bad", "not-tab-separated"),
                ):
                    mol = Chem.MolFromSmiles("CC")
                    mol.SetProp("ID", name)
                    mol.SetProp("FW", "30.0")
                    mol.SetProp("Primaries", shifts)
                    writer.write(mol)
            finally:
                writer.close()

            result = sdf_module.Process_sdf_files(
                [str(sdf_path)], ["Primaries"], 0.0, 100.0, 1, 10
            ).compound_inf_dict
            self.assertEqual(["valid"], list(result))


class Recqc1MatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(RECQC1))
        try:
            cls.matching_module = load_module("recqc1", "MATCHINSTITUTE.py")
        finally:
            sys.path.pop(0)

    def _match(self, match_state, database, green_threshold=5.0, **user_groups):
        class InputData:
            Tertiaries = []
            Secondaries = []
            Primaries = []
            Quaternaries = []
            psq = []
            pt = []
            pstq = []

        input_data = InputData()
        input_data.match_state = match_state
        for group_name, shifts in user_groups.items():
            setattr(input_data, group_name, list(shifts))

        class SdfData:
            compound_inf_dict = {"candidate": database}

        matcher = self.matching_module.ProductBestCompare(
            input_data, SdfData(), green_threshold=green_threshold
        )
        entries = matcher.best_match_data_dict[("candidate", len(database[3]))]
        return {atom_index: values for entry in entries for atom_index, values in entry.items()}

    def test_database_atom_order_matches_original_recqc20_semantics(self):
        matched = self._match(
            4,
            [[0, 1], [10.0, 11.0], ["Primaries", "Primaries"], []],
            Primaries=[10.9, 30.0],
        )

        self.assertEqual(10.9, matched[0][1])
        self.assertEqual(30.0, matched[1][1])

    def test_type_compatible_pass_precedes_global_fallback(self):
        matched = self._match(
            4,
            [[0, 1], [10.0, 20.0], ["Primaries", "Secondaries"], []],
            Primaries=[10.1],
            Tertiaries=[20.1],
        )

        self.assertEqual([10.1, 1], [matched[0][1], matched[0][3]])
        self.assertEqual([20.1, 0], [matched[1][1], matched[1][3]])

    def test_ambiguous_dept_type_is_compatible_in_the_typed_pass(self):
        matched = self._match(
            1,
            [[0, 1], [10.0, 50.0], ["Primaries", "Secondaries"], []],
            pt=[10.2],
            Secondaries=[50.1],
        )

        self.assertEqual(-1, matched[0][3])
        self.assertEqual(1, matched[1][3])

    def test_threshold_changes_atom_state_but_not_original_recqc20_assignment(self):
        database = [[0, 1], [10.0, 11.0], ["Primaries", "Primaries"], []]
        strict = self._match(4, database, green_threshold=0.05, Primaries=[10.9, 30.0])
        loose = self._match(4, database, green_threshold=100.0, Primaries=[10.9, 30.0])

        self.assertEqual([strict[0][1], strict[1][1]], [loose[0][1], loose[1][1]])
        self.assertFalse(strict[0][2])
        self.assertTrue(loose[0][2])

    def test_duplicate_numeric_shifts_remain_two_distinct_peaks(self):
        matched = self._match(
            4,
            [[0, 1], [10.0, 10.1], ["Primaries", "Primaries"], []],
            Primaries=[10.0, 10.0],
        )

        self.assertEqual(2, len(matched))
        self.assertEqual([10.0, 10.0], [matched[0][1], matched[1][1]])


class Recqc1SkeletonScoringTests(unittest.TestCase):
    def test_best_supported_substructure_does_not_depend_on_template_order(self):
        sys.path.insert(0, str(RECQC1))
        try:
            scoring_module = load_module("recqc1", "MATCHINSTITUTE.py")
        finally:
            sys.path.pop(0)

        class EmptyMatches:
            threshold = 1.0
            best_match_data_dict = {}

        scorer = scoring_module.ScoreInstitution(
            EmptyMatches(), threshold_yellow=3.0, max_skeleton_score=0.8
        )
        weak_first = {
            0: [{0: ["red", 4.0]}, {1: ["red", 4.0]}],
            1: [{2: ["green", 0.1]}, {3: ["green", 0.2]}],
        }
        strong_first = {1: weak_first[1], 0: weak_first[0]}

        self.assertEqual([2, 3], scorer.get_final_skeleton_index(weak_first))
        self.assertEqual([2, 3], scorer.get_final_skeleton_index(strong_first))

    def test_core_and_environment_keep_the_published_weighting_layers(self):
        sys.path.insert(0, str(RECQC1))
        try:
            scoring_module = load_module("recqc1", "MATCHINSTITUTE.py")
        finally:
            sys.path.pop(0)

        class EmptyMatches:
            threshold = 1.0
            best_match_data_dict = {}

        scorer = scoring_module.ScoreInstitution(
            EmptyMatches(), threshold_yellow=3.0, max_skeleton_score=0.8
        )
        green_atoms = [
            [0, 10.0, 10.0, 0.0],
            [1, 20.0, 20.0, 0.0],
            [2, 30.0, 30.0, 0.0],
        ]
        core_score, environment_score = scorer.get_compound_score(green_atoms, [], [], 4, [0, 1])
        environment_rate = scorer.none_ske_score_change_rate(4, [0, 1], environment_score)

        self.assertEqual(0.5, core_score)
        self.assertEqual(0.25, environment_score)
        self.assertEqual(1.25, environment_rate)
        self.assertAlmostEqual(0.7, core_score + environment_score / environment_rate)


class SpeedMatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module("speed", "CarbonScoreProcess.py")

    def _matcher(self, merge="N", green=1.0, yellow=10.0):
        matcher = self.module.GreedyMatcher.__new__(self.module.GreedyMatcher)
        matcher.C_merge = merge
        matcher.match_green_threshold = green
        matcher.match_yellow_threshold = yellow
        matcher.match_fine_ranges = None
        return matcher

    @staticmethod
    def _score(delta, green=1.0, yellow=10.0):
        if delta <= green:
            return 1.0
        if delta < yellow:
            return 1.0 - (delta - green) / (yellow - green)
        return 0.0

    def test_speed_maximizes_green_plus_yellow_total_score(self):
        matcher = self._matcher()
        candidates = [
            {"c_index": 0, "c_shift": 0.0},
            {"c_index": 1, "c_shift": 1.0},
        ]
        matches, unmatched = matcher._vectorized_match([1.5, 2.5], candidates)
        assigned = {carbon_index: user_shift for user_shift, carbon_index, _ in matches}

        selected_score = sum(
            self._score(abs(candidates[index]["c_shift"] - assigned[index])) for index in range(2)
        )
        alternative_score = self._score(2.5) + self._score(0.5)

        self.assertEqual([], unmatched)
        self.assertEqual({0: 1.5, 1: 2.5}, assigned)
        self.assertGreater(selected_score, alternative_score)

    def test_speed_preserves_duplicate_peak_instances(self):
        matcher = self._matcher(green=0.5, yellow=2.0)
        candidates = [
            {"c_index": 0, "c_shift": 10.0},
            {"c_index": 1, "c_shift": 10.1},
        ]
        matches, unmatched = matcher._vectorized_match([10.0, 10.0], candidates)

        self.assertEqual(2, len(matches))
        self.assertEqual([], unmatched)

    def test_speed_keeps_unmatched_carbons_as_zero_score_placeholders(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            db_path = Path(temp_dir) / "speed-placeholders.db"
            with closing(sqlite3.connect(db_path)) as conn:
                conn.execute(
                    "CREATE TABLE molecules "
                    "(mol_id INTEGER PRIMARY KEY, name TEXT, carbon_count INTEGER, fw REAL)"
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
                conn.execute("INSERT INTO molecules VALUES (1, 'candidate', 2, 100.0)")
                conn.executemany(
                    """
                    INSERT INTO carbons
                    (mol_id, c_index, h_count, c_shift, h_shifts, connected_carbon)
                    VALUES (1, ?, 0, ?, '[]', '[]')
                    """,
                    ((0, 10.0), (1, 20.0)),
                )
                conn.commit()

            matcher = self.module.GreedyMatcher(
                str(db_path),
                [10.0, 100.0],
                c_range=(2, 2),
                fw_range=(0.0, 200.0),
                mode="untyped",
                C_merge="N",
            )
            matcher.match_green_threshold = 0.5
            matcher.match_yellow_threshold = 2.0
            matcher.run(max_workers=1)

            with closing(sqlite3.connect(db_path)) as conn:
                rows = conn.execute(
                    "SELECT db_c_index, user_shift, type_matched "
                    "FROM greedy_matches ORDER BY db_c_index"
                ).fetchall()

            self.assertEqual([(0, 10.0, 1), (1, None, 0)], rows)

    def test_speed_gui_repairs_missing_windowed_console_streams(self):
        gui_path = VERSIONS["speed"] / "gui.py"
        module_name = "test_speed_gui_console_streams"
        spec = importlib.util.spec_from_file_location(module_name, gui_path)
        module = importlib.util.module_from_spec(spec)
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.path.insert(0, str(VERSIONS["speed"]))
        try:
            sys.stdout = None
            sys.stderr = None
            spec.loader.exec_module(module)
            self.assertIsNotNone(sys.stdout)
            self.assertIsNotNone(sys.stderr)
            sys.stdout.write("")
            sys.stderr.write("")
        finally:
            repaired_stdout, repaired_stderr = sys.stdout, sys.stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            for stream in (repaired_stdout, repaired_stderr):
                if stream not in (original_stdout, original_stderr) and stream is not None:
                    stream.close()
            sys.path.pop(0)


class CarbonScoringBoundaryTests(unittest.TestCase):
    def test_zero_ppm_is_a_real_shift_in_every_2x_version(self):
        for version in VERSIONS:
            with self.subTest(version=version), tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
                db_path = Path(temp_dir) / "zero-shift.db"
                with closing(sqlite3.connect(db_path)) as conn:
                    conn.execute(
                        """
                        CREATE TABLE greedy_matches (
                            mol_id INTEGER,
                            mol_name TEXT,
                            user_shift REAL,
                            db_c_index INTEGER,
                            db_shift REAL,
                            type_matched INTEGER,
                            connected_h_shifts TEXT,
                            connected_carbons TEXT
                        )
                        """
                    )
                    conn.execute(
                        "INSERT INTO greedy_matches VALUES (1, 'zero', 0.0, 0, 0.0, 1, '[]', '[]')"
                    )
                    conn.commit()

                scorer_class = load_module(version, "CarbonScoreProcess.py").CarbonScorer
                scorer = scorer_class(str(db_path))
                scorer.create_results_table()
                scorer.process_scoring(mode="global", global_thresholds=(0.5, 2.0))

                with closing(sqlite3.connect(db_path)) as conn:
                    score, color = conn.execute("SELECT score, color FROM carbon_scores").fetchone()
                self.assertEqual(1.0, score)
                self.assertIn(color.lower(), {"#00ff00", "green"})


class MoleculeDrawingCompatibilityTests(unittest.TestCase):
    def test_current_rdkit_draw_options_work_in_every_2x_version(self):
        for version in VERSIONS:
            with self.subTest(version=version), tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
                module = load_module(version, "CarbonScorerResult.py")
                visualizer = module.CarbonResultVisualizer.__new__(module.CarbonResultVisualizer)
                output = Path(temp_dir) / "molecule.png"
                visualizer._draw_labeled_molecule(
                    mol=Chem.MolFromSmiles("CCO"),
                    highlight_atoms=[0],
                    highlight_colors=[(0.0, 1.0, 0.0)],
                    labels={0: "10.0"},
                    filename=str(output),
                )
                self.assertGreater(output.stat().st_size, 0)

    def test_current_rdkit_draw_options_work_in_recqc1(self):
        sys.path.insert(0, str(RECQC1))
        try:
            module = load_module("recqc1", "DRAWCOMPOUND.py")
        finally:
            sys.path.pop(0)

        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            previous = Path.cwd()
            try:
                os.chdir(temp_dir)
                drawer = module.DrawCompound.__new__(module.DrawCompound)
                image = drawer.draw_compound(
                    green_index=[0],
                    yellow_index=[],
                    orange_index=[],
                    red_index=[],
                    mol=Chem.MolFromSmiles("CCO"),
                    rank=1,
                    color1=(0.0, 1.0, 0.0),
                    color2=(1.0, 1.0, 0.0),
                    color3=(1.0, 0.5, 0.0),
                    color4=(1.0, 0.0, 0.0),
                    data_type=[10.0],
                    num="test",
                    labelsize=0.6,
                    index_num=0,
                )
            finally:
                os.chdir(previous)
            self.assertGreater(len(image), 0)
            self.assertTrue(list((Path(temp_dir) / "RESULT_IMG").glob("*.png")))


if __name__ == "__main__":
    unittest.main()
