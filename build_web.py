"""Assemble everything the browser needs into web/.

    python3 build_web.py --flybody /path/to/flybody/flybody/fruitfly/assets

Produces: fly.glb, skeleton.json, legmap.json, brain.json, policies.json
and (if MuJoCo and trained policies are present) clips.json.
"""
import argparse, json, shutil, subprocess, sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--flybody", required=True, help="flybody's fruitfly/assets directory")
p.add_argument("--clips", action="store_true", help="also record MuJoCo physics clips")
p.add_argument("--clip-seconds", type=float, default=12.0)
a = p.parse_args()

assets = Path(a.flybody)
web = Path("web"); web.mkdir(exist_ok=True)

from flybrain.body.export_body import build
build(str(assets / "fruitfly.xml"), web)

from flybrain.body import legs
legs.export()
print("wrote web/legmap.json")

shutil.copy("data/circuit.json", web / "brain.json")
print("wrote web/brain.json")

policies = {}
for f in sorted(Path("policies").glob("*.json")):
    if f.name == "benchmarks.json":
        continue
    d = json.loads(f.read_text())
    policies[d["task"]] = {"theta": d["theta"], "validation": d["validation"]}
if not policies:
    sys.exit("no trained policies in policies/ -- run: python3 -m flybrain.body.train_body --task all")
(web / "policies.json").write_text(json.dumps(policies, separators=(",", ":")))
print(f"wrote web/policies.json ({', '.join(policies)})")

if a.clips:
    subprocess.check_call([sys.executable, "-m", "flybrain.body.mujoco_clip",
                           "--xml", str(assets / "floor.xml"),
                           "--seconds", str(a.clip_seconds)])
print("\nweb/ is ready. Serve it with:  python3 -m http.server -d web 8000")
