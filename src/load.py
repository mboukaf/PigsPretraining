from pathlib import Path
import h5py
import numpy as np
from utils import find_root

# find_root() from the first cell of main.ipynb
ROOT = find_root()
H5   = ROOT / "data" / "processed" / "pigs_useful.h5"

with h5py.File(H5, "r") as f:
    f.visit(print)                        # every group/dataset in the file
    FS       = int(f.attrs["fs"])
    CHANNELS = list(f.attrs["channels"])
    UNITS    = dict(zip(CHANNELS, f.attrs["units"]))
    PIGS     = sorted(int(k[3:]) for k in f)   # "pig4" → 4

print(FS, PIGS)        # 250 [4, 6, 7, 8]
print(CHANNELS)


def show(name, obj):
    kind  = "GROUP  " if isinstance(obj, h5py.Group) else "DATASET"
    extra = f"  shape={obj.shape} dtype={obj.dtype}" if isinstance(obj, h5py.Dataset) else ""
    print(f"{kind} /{name}{extra}")
    for k, v in obj.attrs.items():
        print(f"    @{k} = {v!r}")

with h5py.File(H5, "r") as f:
    show("", f)              # the root itself (visititems skips it)
    f.visititems(show)       # then every group and dataset below it
