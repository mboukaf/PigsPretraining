"""Constants shared by every preprocessing step (see Docs/embedding_space_procedure.md)."""
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[2]
RAW_DIR  = ROOT / "data" / "osfstorage-archive"
PROC_DIR = ROOT / "data" / "processed"

FS = 250  # Hz, Cushway et al. PLOS ONE 2024

ALL_PIGS = [1, 2, 3, 4, 5, 6, 7, 8]
PAPER_ID = {4: 1, 6: 2, 7: 3, 8: 4}  # pigs used in the paper, with the paper's numbering

# Spec symbol -> name of the field in the raw .mat struct
RAW_NAME = {
    "P_ao":  "aorticPressure",
    "Q_ao":  "aorticFlow",
    "P_fem": "femoralPressure",
    "P_lv":  "leftVentriclePressure",
    "V_lv":  "leftVentricleVolume",
    "P_vc":  "venaCavaPressure",
    "P_aw":  "airwayPressure",
    "Q_aw":  "airwayFlow",
}
CARDIAC = ["P_ao", "Q_ao", "P_fem", "P_lv", "V_lv", "P_vc"]  # encoder inputs (M = 6)
AIRWAY  = ["P_aw", "Q_aw"]                                   # evaluation only
SIGNALS = CARDIAC + AIRWAY

UNITS = {
    "P_ao": "mmHg", "Q_ao": "L/min", "P_fem": "mmHg", "P_lv": "mmHg",
    "V_lv": "mL", "P_vc": "mmHg", "P_aw": "cmH2O", "Q_aw": "mL/s",
}
