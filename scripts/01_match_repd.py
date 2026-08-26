"""Matching BM units to REPD projects to get coordinates and boundary side.

The Elexon registration data has no location, so I match each transmission wind
BM unit to its REPD project by name, convert the OSGB grid reference to
latitude and longitude, and use that to decide which units sit north of B6.

Names do not match exactly on either side, so I normalise both, score the match
and flag anything below the confidence threshold. Corrections then come from a
hand verified overrides file rather than being buried in the code, so every
manual decision stays visible and auditable.

"""

import re
from difflib import SequenceMatcher

import pandas as pd
from pyproj import Transformer

from src import config

# Words that appear on one side but not the other and carry no identifying
# information, so removing them makes the two naming styles comparable
NOISE = [
    "wind farm", "windfarm", "wind park", "offshore wind", "offshore",
    "onshore", "extension", "limited", " ltd", "phase", " wf ", " osp ",
    "east", "west", "north", "south",
]

AUTO_ACCEPT = 0.85
REVIEW_FLOOR = 0.60


def normalise(name: str) -> str:
    """Reducing a site name to its identifying words."""
    text = f" {str(name).lower()} "
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    for word in NOISE:
        text = text.replace(word, " ")
    text = re.sub(r"\b\d+\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def score(left: str, right: str) -> float:
    """Scoring two normalised names, boosting when one contains the other.

    REPD often carries extra words such as the operating company, so a plain
    similarity ratio understates an otherwise obvious match.
    """
    if not left or not right:
        return 0.0
    ratio = SequenceMatcher(None, left, right).ratio()
    if left in right or right in left:
        ratio = max(ratio, 0.9)
    return ratio


def load_repd() -> pd.DataFrame:
    """Loading REPD wind projects that are built or being built."""
    repd = pd.read_csv(config.REPD_PATH, encoding="latin-1", low_memory=False)

    is_wind = repd["Technology Type"].astype(str).str.contains("Wind", case=False)
    is_built = repd["Development Status (short)"].astype(str).str.contains(
        "Operational|Under Construction", case=False, na=False
    )
    repd = repd[is_wind & is_built].copy()

    repd = repd.rename(
        columns={
            "Site Name": "site_name",
            "Installed Capacity (MWelec)": "repd_capacity_mw",
            "X-coordinate": "easting",
            "Y-coordinate": "northing",
            "Country": "country",
        }
    )
    repd["easting"] = pd.to_numeric(repd["easting"], errors="coerce")
    repd["northing"] = pd.to_numeric(repd["northing"], errors="coerce")
    repd = repd.dropna(subset=["easting", "northing"])

    # Trailing spaces appear in some site names, so I strip before matching
    repd["site_name"] = repd["site_name"].astype(str).str.strip()
    repd["name_key"] = repd["site_name"].map(normalise)
    return repd.reset_index(drop=True)


def add_lat_lon(frame: pd.DataFrame) -> pd.DataFrame:
    """Converting OSGB eastings and northings to latitude and longitude."""
    transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(
        frame["easting"].to_numpy(), frame["northing"].to_numpy()
    )
    frame["latitude"] = lat
    frame["longitude"] = lon
    return frame


def best_match(name_key: str, repd: pd.DataFrame) -> tuple[int | None, float]:
    """Finding the highest scoring REPD row for one BM unit name."""
    scores = repd["name_key"].map(lambda candidate: score(name_key, candidate))
    if scores.empty or scores.max() == 0:
        return None, 0.0
    return scores.idxmax(), float(scores.max())


def apply_overrides(lookup: pd.DataFrame, repd: pd.DataFrame) -> pd.DataFrame:
    """Replacing automatic matches with the hand verified ones.

    Most overrides name an exact REPD site, so I pull that row's coordinates
    and country rather than trusting the fuzzy match. Some only correct the
    boundary side of a unit whose site match I am not trying to fix, and those
    leave the site name blank and are handled by the forcing step later.
    """
    if not config.OVERRIDES_PATH.exists():
        print("No overrides file found, skipping")
        return lookup

    overrides = pd.read_csv(config.OVERRIDES_PATH)
    applied = 0

    for row in overrides.itertuples():
        # A blank site name means the override only forces the boundary side
        if pd.isna(row.repd_site_name):
            continue

        site = repd[repd["site_name"] == str(row.repd_site_name).strip()]
        if site.empty:
            print(f"  override site not found in REPD: {row.repd_site_name}")
            continue

        site = site.iloc[0]
        mask = lookup["elexonBmUnit"] == row.elexonBmUnit
        if not mask.any():
            print(f"  override unit not in fleet: {row.elexonBmUnit}")
            continue

        lookup.loc[mask, "repd_site_name"] = site["site_name"]
        lookup.loc[mask, "repd_capacity_mw"] = site["repd_capacity_mw"]
        lookup.loc[mask, "country"] = site["country"]
        lookup.loc[mask, "latitude"] = site["latitude"]
        lookup.loc[mask, "longitude"] = site["longitude"]
        lookup.loc[mask, "match_status"] = "override"
        lookup.loc[mask, "match_note"] = row.note
        applied += 1

    print(f"Applied {applied} site overrides of {len(overrides)} rows")
    return lookup


def main() -> None:
    fleet = pd.read_csv(config.FLEET_PATH).drop_duplicates(subset="elexonBmUnit")
    print(f"Fleet units after dedupe: {len(fleet):,}")

    repd = add_lat_lon(load_repd())
    print(f"REPD wind projects with coordinates: {len(repd):,}")

    # Matching on the BM unit name, since that is closer to the site name than
    # the lead party, which is usually a special purpose company
    fleet["name_key"] = fleet["bmUnitName"].map(normalise)

    rows = []
    for record in fleet.itertuples():
        index, match_score = best_match(record.name_key, repd)
        match = repd.loc[index] if index is not None else None
        rows.append(
            {
                "elexonBmUnit": record.elexonBmUnit,
                "bmUnitName": record.bmUnitName,
                "leadPartyName": record.leadPartyName,
                "capacity_mw": record.generationCapacity,
                "repd_site_name": match["site_name"] if match is not None else None,
                "repd_capacity_mw": match["repd_capacity_mw"] if match is not None else None,
                "country": match["country"] if match is not None else None,
                "latitude": match["latitude"] if match is not None else None,
                "longitude": match["longitude"] if match is not None else None,
                "match_score": round(match_score, 3),
            }
        )

    lookup = pd.DataFrame(rows)

    # Anything below the accept threshold gets looked at by hand rather than
    # trusted, and anything below the review floor counts as no match at all
    lookup["match_status"] = pd.cut(
        lookup["match_score"],
        bins=[-0.01, REVIEW_FLOOR, AUTO_ACCEPT, 1.01],
        labels=["unmatched", "review", "accepted"],
    ).astype(str)
    lookup["match_note"] = ""

    lookup = apply_overrides(lookup, repd)

    # B6 runs along the Scotland England border, so country gives us the split
    # directly. B4 sits further north and needs the latitude.
    lookup["north_of_b6"] = lookup["country"].astype(str).str.strip() == "Scotland"
    lookup["north_of_b4"] = lookup["latitude"] > config.LAT_NORTH_OF_B4

    # A few units connect on the other side of the boundary from where they
    # physically sit, so the overrides file can force the classification
    if config.OVERRIDES_PATH.exists():
        forced = pd.read_csv(config.OVERRIDES_PATH).dropna(subset=["force_north_of_b6"])
        for row in forced.itertuples():
            # Reading the flag as text, since bool("FALSE") is True in Python
            # and would silently invert every forced classification
            value = str(row.force_north_of_b6).strip().upper() == "TRUE"
            mask = lookup["elexonBmUnit"] == row.elexonBmUnit
            lookup.loc[mask, "north_of_b6"] = value

    lookup = lookup.sort_values("capacity_mw", ascending=False).reset_index(drop=True)
    lookup.to_csv(config.BMU_LOOKUP_PATH, index=False)

    print(f"\nMatch status:\n{lookup['match_status'].value_counts().to_string()}")

    usable = lookup["match_status"].isin(["accepted", "override"])
    scottish = lookup[lookup["north_of_b6"] & usable]
    print(f"\nScottish transmission wind units: {len(scottish):,}")
    print(f"Scottish capacity: {scottish['capacity_mw'].sum():,.0f} MW")
    print(f"North of B4: {scottish['north_of_b4'].sum():,} units")

    print(f"\nTop {config.TOP_N_BMUS} Scottish units by capacity:")
    columns = ["elexonBmUnit", "repd_site_name", "capacity_mw", "latitude", "match_status"]
    print(scottish[columns].head(config.TOP_N_BMUS).to_string(index=False))

    print(f"\nWrote {config.BMU_LOOKUP_PATH}")


if __name__ == "__main__":
    main()