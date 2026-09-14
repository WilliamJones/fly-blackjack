"""Watch the trained fly play, in your terminal."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

from .circuit import load, load_silenced
from .envs import Dodger
from .readout import act, unpack, INPUT_GAIN

BAR = " .:-=+*#%@"


def activity_bar(h: np.ndarray) -> str:
    return "".join(BAR[min(9, int(abs(v) * 9.99))] for v in h)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoint.json")
    p.add_argument("--seed", type=int, default=2100001)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--fps", type=float, default=45.0)
    p.add_argument("--silenced", action="store_true",
                   help="run the control condition: circuit contributes nothing")
    a = p.parse_args()

    theta = np.array(json.loads(Path(a.checkpoint).read_text())["theta"])
    circuit = load_silenced() if a.silenced else load()
    env = Dodger(max_seconds=a.seconds)
    obs = env.reset(a.seed); circuit.reset()
    names = ["RUN ", "JUMP", "DUCK"]
    done = False
    while not done:
        h = circuit.step(obs)
        action = act(theta, h, circuit.n_outputs, env.N_ACTIONS)
        obs, score, done = env.step(action)
        sys.stdout.write("\x1b[2J\x1b[H")
        print(env.render())
        print(f"action = {names[action]}   {'(circuit SILENCED)' if a.silenced else ''}")
        print("descending cell activity")
        print("  " + activity_bar(h))
        print("  " + "".join("+" if v >= 0 else "-" for v in h))
        time.sleep(1.0 / a.fps)
    print(f"\nfinished: {env.t:.2f}s, score {score:.1f}")


if __name__ == "__main__":
    main()
