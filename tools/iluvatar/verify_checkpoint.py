"""Verify the published runtime sources; does not import GPU packages."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
manifest = json.loads((root / "docs/iluvatar-minicpm/manifest.json").read_text(encoding="utf-8"))
verified = []
for name, expected in manifest["runtime_file_hashes"].items():
    path = root / "vllm_fl" / name
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise SystemExit("Checkpoint hash mismatch: " + name)
    ast.parse(data, filename=str(path))
    verified.append(name)
print(json.dumps({"historical_checkpoint": manifest["candidate_commit"],
                  "byte_exact_runtime_files": verified, "gpu_work_started": False}, indent=2))
