"""Re-download the bundled 80-cell circuit (22 KB) from its original source.

The kit already ships data/circuit.json. Run this only if you want to refresh it.
To build a DIFFERENT circuit from the full connectome, use build_your_own_circuit.py.
"""
import urllib.request, pathlib, json

URL = ("https://raw.githubusercontent.com/cobanov/flyjump/main/"
       "public/data/connectome/graph.json")
out = pathlib.Path(__file__).parent / "data" / "circuit.json"
out.parent.mkdir(exist_ok=True)
data = urllib.request.urlopen(URL).read()
graph = json.loads(data)
out.write_bytes(data)
print(f"wrote {out} - {len(graph['nodes'])} cells, {len(graph['edges'])} edges")
print("Connectome data: FlyEM / HHMI Janelia MaleCNS v1.0, CC BY 4.0.")
print("Circuit extraction: Mert Cobanov, github.com/cobanov/flyjump")
