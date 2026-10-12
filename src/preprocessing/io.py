"""Reading the raw OSF .mat recordings."""
import numpy as np
import scipy.io as sio

from .config import RAW_DIR, RAW_NAME, SIGNALS


def load_raw(pig_id: int, signals: list[str] = SIGNALS) -> np.ndarray:
    """Raw recording of one pig as float64 (len(signals), N), rows in `signals` order.

    Fields are looked up by name: their order in the .mat struct differs between pigs.
    """
    mat = sio.loadmat(RAW_DIR / f"Pig{pig_id}.mat", squeeze_me=True, struct_as_record=False)
    s = mat[f"Pig{pig_id}"]
    return np.stack([np.asarray(getattr(s, RAW_NAME[sig]), dtype=np.float64) for sig in signals])
