"""Insert rendered result tables for a frozen run between the RESULTS markers of the root README."""
import argparse
import subprocess
import sys
from pathlib import Path

from podium.settings import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--platform", default="TigerGraph Savanna 4.2.5 (TG-00 workspace)")
a = ap.parse_args()
readme = ROOT.parent / "README.md"
text = readme.read_text(encoding="utf-8")
parts = []
for split in ("public", "robustness", "hidden"):
    try:
        out = subprocess.check_output([sys.executable, str(ROOT / "scripts" / "render_results.py"), "--run", a.run, "--split", split], cwd=ROOT).decode()
        parts.append(out.strip())
    except subprocess.CalledProcessError:
        pass
block = f"Frozen run **`{a.run}`** on {a.platform}; answer model `gpt-4.1`, judge `gemini-2.5-flash`, embeddings `text-embedding-3-small`. Artifacts: `Codes/artifacts/runs/{a.run}/` (submission package in `submission/`).\n\n" + "\n\n".join(parts)
start, end = "<!-- RESULTS:BEGIN -->", "<!-- RESULTS:END -->"
i, j = text.index(start) + len(start), text.index(end)
text = text[:i] + "\n" + block + "\n" + text[j:]
text = text.replace("(see the results section above for the id)", f"(`{a.run}`)")
readme.write_text(text, encoding="utf-8")
print("README updated with", a.run)
