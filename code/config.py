"""Portable filesystem configuration; no data are downloaded automatically."""
import os
from pathlib import Path
ROOT = Path(os.environ.get("MOSAIC_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()
JD = str(ROOT)
DTA = str(Path(os.environ.get("MOSAIC_DATA_PATH", ROOT / "data" / "Base Ocupacional.dta")).expanduser().resolve())
