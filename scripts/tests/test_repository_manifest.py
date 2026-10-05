"""Exercise release boundaries against real, isolated filesystem trees."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "repository_manifest.py"


class RepositoryManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(SCRIPT.is_file(), "release collector must exist")
        spec = importlib.util.spec_from_file_location("repository_manifest", SCRIPT)
        assert spec is not None and spec.loader is not None
        self.manifest = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.manifest)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def put(self, relative: str, content: bytes = b"abc") -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def paths(self) -> list[str]:
        return [path.relative_to(self.root).as_posix() for path in self.manifest.collect_files(self.root)]

    def test_collects_runtime_sources_licenses_tests_and_prebuilt_ui(self) -> None:
        # Omitting the result schema or prebuilt browser code breaks a fresh checkout.
        allowed = [
            ".gitattributes", ".gitignore", "README.md", "SOURCE_MANIFEST.json",
            "THIRD_PARTY_NOTICES.md", "assets.lock.json", "pyproject.toml", "uv.lock",
            "backend/inspection_demo/main.py", "backend/inspection_demo/expected_assets.json",
            "backend/tests/test_api_hf.py", "docs/NOTEBOOK_QUICKSTART.md",
            "frontend/src/App.test.tsx", "frontend/src/test/setup.ts",
            "frontend/dist/index.html", "frontend/dist/assets/app.js",
            "frontend/dist/assets/app.css", "frontend/dist/assets/font.woff2",
            "scripts/demo_launcher.py", "scripts/notebook_client.py",
            "scripts/repository_manifest.py", "scripts/run_local_server.py",
            "scripts/serve_local.py", "scripts/tests/test_launcher_lifecycle.py",
            "scripts/verify_release.py", "scripts/upload_existing_space.py",
            "scripts/deploy_hf.ps1", "scripts/tests/test_demo_launcher.py",
            "scripts/tests/test_notebook_client.py", "scripts/tests/test_repository_manifest.py",
            "vendor/cobbie-ecore/LICENSE", "vendor/cobbie-ecore/README.md",
            "vendor/cobbie-ecore/src/config.py", "vendor/cobbie-ecore/src/schemas/result.py",
            "vendor/cobbie-ecore/src/baml/baml_client/types.py",
            "vendor/cobbie-ecore/src/baml/baml_src/tog.baml",
            "vendor/tog-ifc-ecore/LICENSE", "vendor/tog-ifc-ecore/pyproject.toml",
            "vendor/tog-ifc-ecore/src/tog/engine.py", "vendor/tog-ifc-ecore/src/tog/py.typed",
            "vendor/text-gnn-plugin/pyproject.toml",
        ]
        for relative in reversed(allowed):
            self.put(relative)
        self.assertEqual(self.paths(), sorted(allowed))

    def test_unreviewed_files_secrets_and_runtime_data_never_enter_release(self) -> None:
        # Broad globbing under source directories would leak keys, caches, or run data.
        self.put("README.md")
        denied = [
            ".env", ".env.example", ".git/config", ".tmp/main.py",
            ".runtime/model.py", ".runtime-host/key.py", ".webapp/state.json",
            "assets/ecore.ifc", "checkpoints/model.pt", "data/response.json",
            "frontend/node_modules/package/index.js", "frontend/dist/assets/app.js.map",
            "frontend/dist/assets/private.key", "frontend/src/.env.local",
            "frontend/src/data/response.json", "frontend/src/cache/cached.ts",
            "frontend/src/.runtime-session/cached.ts", "frontend/src/secret.pem",
            "frontend/src/credentials.json", "frontend/src/private_key.ts",
            "backend/inspection_demo/__pycache__/main.py", "backend/inspection_demo/main.pyc",
            "backend/inspection_demo/data/saved.py", "backend/inspection_demo/.venv/secret.py",
            "backend/inspection_demo/.demo-state/saved.py",
            "backend/tests/.pytest_cache/test_secret.py", "backend/tests/responses.json",
            "docs/private-notes.md", "notebooks/.ipynb_checkpoints/Launch_Demo-checkpoint.ipynb",
            "notebooks/output.log", "scripts/browser_live_replay.py",
            "scripts/prepare_hf_assets.py", "scripts/tests/test_local_secrets.py",
            "vendor/cobbie-ecore/src/db/db.db", "vendor/cobbie-ecore/src/sitecustomize.py",
            "vendor/cobbie-ecore/src/util/python_executor.py",
            "vendor/tog-ifc-ecore/build/lib/tog/engine.py",
            "vendor/tog-ifc-ecore/src/tog_ifc.egg-info/PKG-INFO",
            "vendor/tog-ifc-ecore/src/tog/cache/index.py",
            "vendor/text-gnn-v3.1/src/text_gnn_v5/training.py", "RELEASE_MANIFEST.json",
        ]
        for relative in denied:
            self.put(relative)
        self.assertEqual(self.paths(), ["README.md"])

    def test_nested_git_checkout_is_excluded_even_under_an_allowed_source_tree(self) -> None:
        # A nested checkout must not silently vendor unrelated repository contents.
        self.put("backend/inspection_demo/main.py")
        self.put("backend/inspection_demo/nested/.git/config")
        self.put("backend/inspection_demo/nested/secret.py")
        self.put("frontend/src/worktree/.git", b"gitdir: /elsewhere")
        self.put("frontend/src/worktree/private.ts")
        self.assertEqual(self.paths(), ["backend/inspection_demo/main.py"])

    def test_symlink_cannot_import_a_file_outside_the_project(self) -> None:
        # Following a symlink would let a harmless source name expose an external file.
        target = self.put("private.txt")
        link = self.root / "backend/inspection_demo/main.py"
        link.parent.mkdir(parents=True)
        try:
            link.symlink_to(target)
        except OSError as error:
            self.skipTest(f"Host cannot create test symlinks: {error}")
        with self.assertRaisesRegex(ValueError, "[Ss]ymlink|junction"):
            self.manifest.collect_files(self.root)

    def test_symlinked_source_directory_is_rejected(self) -> None:
        # Checking only leaf files would still allow a linked package directory.
        target = self.root / "external"
        target.mkdir()
        (target / "main.py").write_bytes(b"private")
        link = self.root / "backend"
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"Host cannot create test symlinks: {error}")
        with self.assertRaisesRegex(ValueError, "[Ss]ymlink|junction"):
            self.manifest.collect_files(self.root)

    @unittest.skipUnless(os.name == "nt", "Windows junction boundary")
    def test_windows_directory_junction_is_rejected(self) -> None:
        # Junctions bypass is_symlink(), so they need their own boundary check.
        target = self.root / "external"
        target.mkdir()
        (target / "main.py").write_bytes(b"private")
        link = self.root / "backend"
        created = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True, text=True,
        )
        self.assertEqual(created.returncode, 0, created.stdout + created.stderr)
        with self.assertRaisesRegex(ValueError, "[Ss]ymlink|junction"):
            self.manifest.collect_files(self.root)

    def test_manifest_hashes_file_bytes_with_portable_paths_and_excludes_itself(self) -> None:
        # Hashing paths or including the output would make inventories wrong or unstable.
        self.put("README.md", b"abc")
        self.put(".git/config", b"private")
        output = self.root / "RELEASE_MANIFEST.json"
        first = self.manifest.write_manifest(self.root, output)
        self.assertEqual(first["files"], [{
            "path": "README.md", "bytes": 3,
            "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        }])
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), first)
        original = output.read_bytes()
        self.manifest.write_manifest(self.root, output)
        self.assertEqual(output.read_bytes(), original)
        self.assertNotIn(str(self.root), output.read_text(encoding="utf-8"))

    def test_write_does_not_overwrite_a_release_source(self) -> None:
        # A mistaken --output must not replace the README or another selected input.
        readme = self.put("README.md", b"keep")
        with self.assertRaises(ValueError):
            self.manifest.write_manifest(self.root, readme)
        self.assertEqual(readme.read_bytes(), b"keep")

    def test_write_rejects_outputs_outside_the_project(self) -> None:
        # A lexical prefix check can allow ../ to escape the intended output root.
        output = self.root / ".." / (self.root.name + "-outside.json")
        self.addCleanup(output.unlink, missing_ok=True)
        with self.assertRaises(ValueError):
            self.manifest.write_manifest(self.root, output)
        self.assertFalse(output.exists())

    def test_write_does_not_modify_git_metadata(self) -> None:
        # An output override must never replace repository metadata.
        config = self.put(".git/config", b"keep")
        with self.assertRaises(ValueError):
            self.manifest.write_manifest(self.root, config)
        self.assertEqual(config.read_bytes(), b"keep")

    def test_cli_check_detects_changed_missing_and_new_release_files(self) -> None:
        # A check must fail on stale inventories rather than silently rebuilding them.
        readme = self.put("README.md", b"abc")
        command = [sys.executable, str(SCRIPT), "--project-root", str(self.root)]
        written = subprocess.run(command + ["--write"], capture_output=True, text=True)
        self.assertEqual(written.returncode, 0, written.stderr)
        checked = subprocess.run(command + ["--check"], capture_output=True, text=True)
        self.assertEqual(checked.returncode, 0, checked.stderr)
        for change in ("changed", "missing", "new"):
            with self.subTest(change=change):
                if change == "changed":
                    readme.write_bytes(b"changed")
                elif change == "missing":
                    readme.unlink()
                else:
                    readme.write_bytes(b"abc")
                    self.put("backend/inspection_demo/main.py")
                result = subprocess.run(command + ["--check"], capture_output=True, text=True)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)

    def test_cli_list_emits_only_relative_selected_paths(self) -> None:
        # A staging list must not contain diagnostics, absolute paths, or private files.
        self.put("README.md")
        self.put(".env")
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--project-root", str(self.root), "--list"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["README.md"])


if __name__ == "__main__":
    unittest.main()
