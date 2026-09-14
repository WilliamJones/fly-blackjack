"""Fetch the MaleCNS connectivity and verify it is what we think it is.

Not the 1.2 GB Google Storage download. These are preprocessed matrices from
YijieYin/connectome_data_prep -- 107 MB total, same underlying data (Berg et al.
2025 / Nern et al. 2024, CC BY 4.0), reachable anywhere GitHub is.

Every number in EXPECTED was measured from the files, not estimated. If a check
fails the data changed underneath us and nothing downstream should be trusted.
"""
from __future__ import annotations
import sys, time, urllib.error, urllib.request
from pathlib import Path

BASE = ("https://raw.githubusercontent.com/YijieYin/connectome_data_prep/"
        "main/data/maleCNS/")
FILES = {
    "mcns_syncount_all_neuron.npz": 78_770_675,
    "mcns_all_neuron_meta.csv": 28_023_979,
}
DATA = Path("data/malecns")

EXPECTED = {
    "neurons": 161_429,
    "connections": 25_083_972,
    "synapses": 121_933_227,
    "kenyon_cells": 4_064,
    "mbons": 97,
    "pam": 316,
    "ppl1": 16,
    "projection_neurons": 682,
}


CHUNK = 1 << 20

CERT_HELP = """
If the error mentions certificates: the python.org macOS build does not use the
system certificate store until you tell it to. Run this once --

  open "/Applications/Python 3.11/Install Certificates.command"

-- or use whichever version number matches your Python.
"""


def _is_certificate_problem(reason) -> bool:
    text = str(reason).lower()
    return "certificate" in text or "ssl" in text


def _curl(url: str, target: Path, size: int) -> bool:
    """Fall back to curl, which uses the operating system's TLS stack.

    Worth trying whenever urllib fails: a Python install with no certificate
    bundle cannot fetch anything over HTTPS, but curl on the same machine can.
    """
    import shutil, subprocess
    if not shutil.which("curl"):
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["curl", "-fL", "-C", "-", "--retry", "5",
                        "--retry-delay", "2", "-o", str(target), url],
                       check=True)
    except (subprocess.CalledProcessError, OSError):
        return False
    ok = target.exists() and target.stat().st_size == size
    if ok:
        print(f"  got {target.name} via curl")
    return ok


def _fetch_one(url: str, target: Path, size: int, attempts: int = 6) -> None:
    """Download with resume. Large files over plain HTTP get truncated often
    enough that retrying from scratch is not good enough -- we ask for the rest
    with a Range header and append."""
    last_error = "unknown"
    for attempt in range(1, attempts + 1):
        have = target.stat().st_size if target.exists() else 0
        if have == size:
            return
        if have > size:                       # corrupt leftover, start again
            target.unlink()
            have = 0
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                # a server that ignores Range replies 200 and sends the whole file
                mode = "ab" if (have and r.status == 206) else "wb"
                if mode == "wb":
                    have = 0
                with open(target, mode) as f:
                    while True:
                        chunk = r.read(CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
                        if sys.stdout.isatty():
                            print(f"\r  {target.name}  {have/1e6:6.1f} /"
                                  f" {size/1e6:.0f} MB  {100*have/size:5.1f}%",
                                  end="", flush=True)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            reason = getattr(e, "reason", e)
            last_error = f"{type(e).__name__}: {reason}"
            print(f"\n  interrupted -- {last_error}", flush=True)
            if _is_certificate_problem(reason):
                print("  (this looks like a missing certificate bundle;"
                      " trying curl instead)", flush=True)
                if _curl(url, target, size):
                    return
                raise SystemExit(CERT_HELP)
        got = target.stat().st_size if target.exists() else 0
        if got == size:
            if sys.stdout.isatty():
                print()
            return
        if attempt < attempts:
            time.sleep(min(2 ** attempt, 20))
    if _curl(url, target, size):          # last resort: the system's own TLS stack
        return
    raise SystemExit(
        f"could not download {target.name}: got {got:,} of {size:,} bytes after "
        f"{attempts} attempts.\n  last error: {last_error}\n\n"
        f"Fetch the two files by hand instead:\n\n"
        f"  mkdir -p {DATA}\n"
        + "".join(f"  curl -L -C - -o {DATA}/{n} \\\n       {BASE}{n}\n"
                  for n in FILES)
        + f"\nthen re-run this command to verify them.\n{CERT_HELP}")


def download(force: bool = False) -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    for name, size in FILES.items():
        target = DATA / name
        if force and target.exists():
            target.unlink()
        if target.exists() and target.stat().st_size == size:
            print(f"  have {name} ({size/1e6:.0f} MB)")
            continue
        _fetch_one(BASE + name, target, size)
        print(f"  got {name} ({target.stat().st_size/1e6:.0f} MB)")
    return DATA


def load():
    """Return (synapse-count matrix, metadata). Row i of M is meta.idx == i."""
    import pandas as pd, scipy.sparse as sp
    for name, size in FILES.items():
        f = DATA / name
        if not f.exists() or f.stat().st_size != size:
            raise SystemExit(
                f"{f} is missing or incomplete.\n"
                f"Run:  python3 -m flybrain.mb.fetch")
    M = sp.load_npz(DATA / "mcns_syncount_all_neuron.npz").tocsr()
    meta = pd.read_csv(DATA / "mcns_all_neuron_meta.csv", low_memory=False)
    return M, meta


def verify(quiet: bool = False) -> bool:
    import numpy as np
    M, meta = load()
    t = meta["type"].fillna("")
    got = {
        "neurons": M.shape[0],
        "connections": M.nnz,
        "synapses": int(M.sum()),
        "kenyon_cells": int(t.str.match(r"^KC").sum()),
        "mbons": int(t.str.match(r"^MBON").sum()),
        "pam": int(t.str.match(r"^PAM").sum()),
        "ppl1": int(t.str.match(r"^PPL1").sum()),
        "projection_neurons": int((meta["class"].fillna("") == "ALPN").sum()),
    }
    ok = True
    for key, want in EXPECTED.items():
        good = got[key] == want
        ok &= good
        if not quiet:
            print(f"  {'ok ' if good else 'BAD'} {key:20s} {got[key]:>12,}"
                  + ("" if good else f"   expected {want:,}"))
    return bool(ok)


if __name__ == "__main__":
    download(force="--force" in sys.argv)
    print("\nverifying:")
    sys.exit(0 if verify() else 1)
