"""Build and import the distribution outside the editable source checkout."""

from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile


class TestDistribution(unittest.TestCase):
    @unittest.skipUnless(shutil.which("uv"), "distribution test requires uv")
    def test_built_distributions_include_and_import_runtime_modules(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="ferminet-package-test-") as directory:
            output = Path(directory)
            built = subprocess.run(
                [shutil.which("uv"), "build", "--out-dir", str(output)],
                cwd=root, capture_output=True, text=True, encoding="utf-8", timeout=120,
            )
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            required = {path.name for path in (root / "ferminet_pytorch").glob("*.py")}
            with tarfile.open(next(output.glob("*.tar.gz"))) as archive:
                files = {Path(name).name for name in archive.getnames()}
                self.assertTrue(required <= files, required - files)
            installed = output / "installed"
            with zipfile.ZipFile(next(output.glob("*.whl"))) as archive:
                files = {Path(name).name for name in archive.namelist()}
                self.assertTrue(required <= files, required - files)
                self.assertNotIn("envelope_modern.py", files)
                self.assertNotIn("legacy_adapters.py", files)
                self.assertTrue(all(not name.startswith(("tests/", "scripts/", "examples/"))
                                    for name in archive.namelist()))
                archive.extractall(installed)
            code = (
                "import pathlib, sys; sys.path.insert(0, sys.argv[1]); "
                "import ferminet_pytorch as f; import importlib, pkgutil; "
                "assert pathlib.Path(f.__file__).resolve().parent == "
                "pathlib.Path(sys.argv[1]).resolve() / 'ferminet_pytorch'; "
                "assert f.FermiNet and f.KatoCuspJastrow and f.SpinProjector; "
                "assert len(f.__path__) == 1; "
                "[importlib.import_module(m.name) for m in pkgutil.iter_modules(f.__path__, f.__name__ + '.')]; "
                "print('isolated distribution import OK')"
            )
            imported = subprocess.run(
                [sys.executable, "-I", "-c", code, str(installed)],
                cwd=output, capture_output=True, text=True, encoding="utf-8", timeout=60,
            )
            self.assertEqual(imported.returncode, 0, imported.stdout + imported.stderr)


if __name__ == "__main__":
    unittest.main()
