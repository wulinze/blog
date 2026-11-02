"""Check chapter coverage, independent CPU demos, assets and downloadable ZIP consistency."""

import ast
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "source/downloads/leetcuda"
manifest = json.loads((ROOT / "docs/learning/leetcuda-source-map.json").read_text())
new = [t for t in manifest["topics"] if t["source_chapters"][0] >= 4]
assert [t["source_chapters"][0] for t in new] == list(range(4, 41))
assert [t["blog_number"] for t in new] == list(range(5, 42))
assert len({t["slug"] for t in new}) == 37
for p in DATA.rglob("*.py"):
    ast.parse(p.read_text())
assert not list(DATA.rglob("*.pyc")), "Do not publish Python cache files"
steps = 0
for t in new:
    n = t["source_chapters"][0]
    folder = DATA / f"{n:02}"
    scenarios = json.loads((folder / "scenarios.json").read_text())
    article = (ROOT / "source/_posts" / t["article"]).read_text()
    assert "  - LLM 推理" in article and "data-leetcuda-lab" in article
    for case in range(4):
        p = subprocess.run(
            [sys.executable, str(folder / "demo.py"), "--case", str(case)], capture_output=True, text=True, check=True
        )
        assert json.loads(p.stdout) == scenarios[case], (n, case, "snapshot mismatch")
        steps += len(scenarios[case]["steps"])
    for suffix in ["svg", "excalidraw"]:
        asset = ROOT / "source/images/leetcuda" / f"{t['slug']}.{suffix}"
        assert asset.stat().st_size > 1000
    scene = json.loads((ROOT / "source/images/leetcuda" / f"{t['slug']}.excalidraw").read_text())
    assert scene["type"] == "excalidraw" and len(scene["elements"]) > 10
    with zipfile.ZipFile(folder / f"{t['slug']}.zip") as bundle:
        assert bundle.testzip() is None
        for name in bundle.namelist():
            expected = folder / name if (folder / name).exists() else DATA / name
            assert bundle.read(name) == expected.read_bytes(), (n, name, "stale zip")
print(f"PASS: 37 chapters, 148 independent CPU executions, {steps} snapshots, editable images and ZIP bytes")
