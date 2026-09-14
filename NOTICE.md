# Attribution

**Connectome data** — MaleCNS v1.0, FlyEM / HHMI Janelia and collaborators
(Cambridge, MRC LMB, Google Research). Licensed **CC BY 4.0**.
https://male-cns.janelia.org/ · Berg et al., *Cell* (2026).
Any published use must cite it.

**`data/circuit.json`** — the bundled 80-cell subgraph was extracted from the
data above by Mert Cobanov for
[Fly Dino / flyjump](https://github.com/cobanov/flyjump), built with
[fly-connectome-template](https://github.com/cobanov/fly-connectome-template).
The extraction procedure is reproduced in `build_your_own_circuit.py` so you can
generate your own from the raw CC BY tables.

**Dynamics, environment, training code** — written for this kit. The game is
clean-room, not derived from Chromium's Dino.

**flybody** — the anatomically detailed MuJoCo fruit-fly body, by Turaga Lab /
Google DeepMind / HHMI Janelia, Apache 2.0.
https://github.com/TuragaLab/flybody · Vaxenburg et al., *Nature* (2025).
`web/fly.json` and `web/skeleton.json` are decimated geometry and extracted
kinematics derived from that model; it is not redistributed here in full.

**three.js** — MIT, loaded from cdnjs at runtime, not bundled.

**MaleCNS connectivity mirror** — preprocessed matrices from
[YijieYin/connectome_data_prep](https://github.com/YijieYin/connectome_data_prep),
derived from MaleCNS (Berg et al. 2025; Nern et al. 2024). The underlying data is
CC BY 4.0 and must be cited; `data/malecns/` is downloaded, never redistributed here.
