"""Building the GB transmission connected wind fleet from Elexon registration data.

The registration data tells us which units are wind and which are transmission
connected, but not where they sit. The gspGroupId field is a supplier side
field and comes back empty for every transmission generator, so location has
to come from elsewhere. This script writes the full fleet, and the next one
joins REPD coordinates to decide which units sit north of the B6 boundary.
"""

import pandas as pd

from src import config
from src.ingest import elexon


def main() -> None:
    print("Fetching BM unit registration data...")
    units = elexon.fetch_bmunits()
    print(f"Returned {len(units):,} units")

    # The T_ prefix marks transmission connected units, which is our scope.
    # Embedded wind never appears in the Balancing Mechanism, so it can never
    # show curtailment and would only pad the list.
    is_wind = units["fuelType"].astype(str).str.upper() == "WIND"
    is_transmission = units["elexonBmUnit"].astype(str).str.startswith("T_")

    fleet = units[is_wind & is_transmission].copy()
    fleet["generationCapacity"] = pd.to_numeric(
        fleet["generationCapacity"], errors="coerce"
    )
    fleet = fleet.sort_values("generationCapacity", ascending=False)

    columns = ["elexonBmUnit", "bmUnitName", "leadPartyName", "generationCapacity"]
    fleet = fleet[columns].reset_index(drop=True)
    fleet.to_csv(config.FLEET_PATH, index=False)

    print(f"\nTransmission wind units: {len(fleet):,}")
    print(f"Total capacity: {fleet['generationCapacity'].sum():,.0f} MW")
    print(f"Wrote {config.FLEET_PATH}")
    print("\nLargest 30 units:")
    print(fleet.head(30).to_string(index=False))


if __name__ == "__main__":
    main()