"""Central settings for the curtailment forecasting project.

Every module reads paths, dates and the forecast cutoff from here so the
prediction contract stays in one place.
"""

from datetime import time
from pathlib import Path

# Resolving the project root from this file so paths work from any directory
ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
REFERENCE = DATA / "reference"

for _folder in (RAW, INTERIM, PROCESSED, REFERENCE):
    _folder.mkdir(parents=True, exist_ok=True)

# Training window, three years so we cover three winters
DATE_START = "2023-04-01"
DATE_END = "2026-03-31"

# The prediction contract. A forecast for day D is issued at this time on D-1
# and may only use data published before it. Every feature is checked against
# this in tests/test_leakage.py.
FORECAST_CUTOFF_UTC = time(11, 0)

# Elexon Insights base URL, public and needing no API key
ELEXON_BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"

# Boundary attribution by latitude. B6 runs roughly along the Scotland England
# border and B4 sits around the SSEN to SPT interface. The gspGroupId field in
# the Elexon registration data is a supplier side field and comes back empty
# for every transmission generator, so location has to come from REPD instead.
# These thresholds are approximate and get confirmed during the hand check of
# the largest units.
LAT_NORTH_OF_B6 = 55.0
LAT_NORTH_OF_B4 = 56.2

# Keeping only the largest units, since curtailed volume is heavily concentrated
TOP_N_BMUS = 25

# Transmission boundaries we model
BOUNDARIES = ["B4", "B6"]

# Reference files that are version controlled rather than downloaded
FLEET_PATH = REFERENCE / "transmission_wind_units.csv"
BMU_LOOKUP_PATH = REFERENCE / "bmu_boundary_lookup.csv"

REPD_PATH = RAW / "repd.csv"

OVERRIDES_PATH = REFERENCE / "bmu_overrides.csv"

# The per BMU endpoints reject ranges wider than a week, returning 400
CHUNK_DAYS = 7

# Parallel requests during the backfill, kept low to stay polite to the API
MAX_WORKERS = 4