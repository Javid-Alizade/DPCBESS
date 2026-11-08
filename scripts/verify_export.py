"""Check that this folder matches an original copy of the project in everything except text labels.

Run:  python scripts/verify_export.py <original folder> [--manifest <sha256 manifest of the original files>]

Checks:
1. results/ and configs/: every JSON file has the same structure and the same numbers, booleans and nulls as the original;
   only string values may differ, and each differing string is listed. Every other file there is byte-identical.
2. src/, scripts/, tests/: the code structure (Python AST with all string literals blanked) is identical.
3. With --manifest: the original files still have the sha256 recorded in the manifest; files the manifest lists as
   intentionally excluded are absent here; every other original result/config file is present; the run index differs from
   the original only by the rows of excluded runs.
Exit code 0 only if every check passes.
"""
import ast
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def compare_json(a, b, where, texts):
    """Walk two JSON values; return a list of non-text differences and collect text differences."""
    if isinstance(a, dict) and isinstance(b, dict):
        if list(a) != list(b):
            return [f"{where}: keys differ"]
        return [d for k in a for d in compare_json(a[k], b[k], f"{where}/{k}", texts)]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{where}: list length differs"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in compare_json(x, y, f"{where}[{i}]", texts)]
    if isinstance(a, str) and isinstance(b, str):
        if a != b:
            texts.append(f"{where}: {a!r} -> {b!r}")
        return []
    if type(a) is not type(b) or (a != b and not (a != a and b != b)):  # NaN equals NaN here
        return [f"{where}: {a!r} != {b!r}"]
    return []


def blank_strings(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = ""
    return ast.dump(tree)


def main():
    original = Path(sys.argv[1]).resolve()
    manifest = None
    if "--manifest" in sys.argv:
        manifest = json.loads(Path(sys.argv[sys.argv.index("--manifest") + 1]).read_text())
    removed_rows = set(manifest.get("removed_index_rows", [])) if manifest else set()
    failures, texts, n_json, n_bytes, n_code = [], [], 0, 0, 0
    for folder in ("results", "configs"):
        for path in sorted((HERE / folder).rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(HERE).as_posix()
            other = original / rel
            if not other.exists():
                failures.append(f"{rel}: missing in the original")
                continue
            if path.suffix == ".json":
                n_json += 1
                failures += compare_json(json.loads(other.read_text(encoding="utf-8")),
                                         json.loads(path.read_text(encoding="utf-8")), rel, texts)
            else:
                n_bytes += 1
                mine, theirs = path.read_bytes(), other.read_bytes()
                if mine != theirs:
                    expected = b"".join(l for l in theirs.splitlines(keepends=True)
                                        if l.split(b",")[0].decode(errors="replace") not in removed_rows)
                    if not (path.suffix == ".csv" and removed_rows and mine == expected):
                        failures.append(f"{rel}: bytes differ")
    for folder in ("src", "scripts", "tests"):
        for path in sorted((HERE / folder).rglob("*.py")):
            rel = path.relative_to(HERE).as_posix()
            other = original / rel
            if not other.exists():
                if rel != "scripts/verify_export.py":
                    failures.append(f"{rel}: missing in the original")
                continue
            n_code += 1
            if blank_strings(path) != blank_strings(other):
                failures.append(f"{rel}: code structure differs")
    if manifest:
        excluded = manifest.get("excluded", {})
        for rel, digest in {**manifest["files"], **excluded}.items():
            if hashlib.sha256((original / rel).read_bytes()).hexdigest() != digest:
                failures.append(f"original {rel}: sha256 differs from the manifest")
        for rel in excluded:
            if (HERE / rel).exists():
                failures.append(f"{rel}: listed as excluded but present")
        for folder in ("results", "configs"):
            for path in (original / folder).rglob("*"):
                rel = path.relative_to(original).as_posix()
                if path.is_file() and rel in manifest["files"] and not (HERE / rel).exists():
                    failures.append(f"{rel}: missing here")
        print(f"manifest: {len(manifest['files'])} original files checked, {len(excluded)} intentionally excluded "
              f"({len(removed_rows)} run folders, their rows removed from the run index)")
    print(f"JSON files compared: {n_json}; other result/config files compared byte by byte: {n_bytes}; "
          f"Python files compared by structure: {n_code}")
    print(f"text labels that differ: {len(texts)}")
    for t in texts:
        print("  " + t)
    print(f"failures: {len(failures)}")
    for f in failures:
        print("  " + f)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
