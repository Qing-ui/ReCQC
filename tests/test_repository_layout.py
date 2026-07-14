import ast
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APPLICATIONS = (ROOT / "apps" / "recqc-2.0", ROOT / "apps" / "recqc-hungarian")
RECQC_1_APPLICATION = ROOT / "apps" / "recqc-1.0"
CORE_MODULES = {
    "CarbonScoreProcess.py",
    "CarbonScorerResult.py",
    "CombineScorerResult.py",
    "HSQCScoreProcess.py",
    "HSQCScorerResult.py",
    "PROCESSSDFFILES.py",
    "SDFFileSelector.py",
    "gui.py",
}


class RepositoryLayoutTests(unittest.TestCase):
    def test_required_repository_files_exist(self):
        for relative_path in (
            "README.md",
            "CHANGELOG.md",
            "CITATION.cff",
            "CONTRIBUTING.md",
            "LICENSE",
            "SECURITY.md",
            "environment.yml",
            "requirements.txt",
        ):
            with self.subTest(path=relative_path):
                self.assertTrue((ROOT / relative_path).is_file())

    def test_both_current_applications_have_core_modules(self):
        for application in APPLICATIONS:
            with self.subTest(application=application.name):
                self.assertTrue(application.is_dir())
                self.assertEqual(CORE_MODULES, {path.name for path in application.glob("*.py")})
                self.assertTrue((application / "README.md").is_file())

    def test_current_python_sources_parse(self):
        for application in (*APPLICATIONS, RECQC_1_APPLICATION):
            for source_path in application.glob("*.py"):
                with self.subTest(source=source_path.relative_to(ROOT)):
                    ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))

    def test_recqc_1_application_is_documented(self):
        self.assertTrue((RECQC_1_APPLICATION / "README.md").is_file())
        self.assertTrue((RECQC_1_APPLICATION / "requirements.txt").is_file())
        self.assertTrue((RECQC_1_APPLICATION / "gui.py").is_file())

    def test_generated_artifacts_are_not_tracked(self):
        forbidden_parts = {"__pycache__", "build", "dist", "results"}
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        tracked_paths = [Path(path) for path in result.stdout.splitlines()]
        generated_paths = [
            path for path in tracked_paths if forbidden_parts.intersection(path.parts)
        ]
        self.assertFalse(generated_paths, "Generated directories must not be committed.")


if __name__ == "__main__":
    unittest.main()
