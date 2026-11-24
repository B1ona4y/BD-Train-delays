"""Quick local overview of the raw Parquet data (pandas + pyarrow, no Spark).

Usage:
    python scripts/inspect_data.py                 # all files in data/dirty
    python scripts/inspect_data.py path/to/dir_or_file.parquet
"""

import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

DEFAULT_DIR = Path(__file__).resolve().parents[1] / "data" / "dirty"
CATEGORICAL = ["provider", "route_id", "trip_status", "stop_status", "delay_source", "timetable_version"]
DELAYS = ["arrival_delay_min", "departure_delay_min"]

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


def section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def list_files(target: Path) -> list[Path]:
    files = sorted(target.glob("*.parquet")) if target.is_dir() else [target]
    if not files:
        sys.exit(f"No .parquet files found in {target}")
    return files


def check_schemas(files: list[Path]) -> None:
    section("FILES & SCHEMA")
    ref = pq.read_schema(files[0])
    print(ref)
    print()
    for f in files:
        meta = pq.ParquetFile(f).metadata
        same = "same schema" if pq.read_schema(f).equals(ref) else "SCHEMA DIFFERS!"
        print(f"{f.name:32} rows={meta.num_rows:>8,}  size={f.stat().st_size / 1e6:5.1f} MB  {same}")


def overview(df: pd.DataFrame) -> None:
    section(f"TOTAL: {len(df):,} rows x {df.shape[1]} columns")
    print(df.head(5).T)

    section("COLUMNS: dtype / nulls / unique / example")
    stats = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "nulls": df.isna().sum(),
        "null_%": (df.isna().mean() * 100).round(1),
        "unique": df.nunique(),
        "example": df.apply(lambda c: c.dropna().iloc[0] if c.notna().any() else None),
    })
    print(stats)


def categorical(df: pd.DataFrame) -> None:
    section("CATEGORICAL VALUES (top 15)")
    for col in CATEGORICAL:
        if col in df:
            print(f"\n--- {col} ({df[col].nunique()} unique) ---")
            print(df[col].value_counts(dropna=False).head(15).to_string())


def delays(df: pd.DataFrame) -> None:
    section("DELAYS (minutes)")
    print(df[DELAYS].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.9, 0.99]).round(2))

    d = df["arrival_delay_min"]
    classes = pd.cut(d, bins=[-float("inf"), 3, 10, float("inf")], labels=["0-3", "4-10", ">10"])
    print("\nTarget classes (arrival_delay_min):")
    print(classes.value_counts(dropna=False, normalize=True).mul(100).round(1).to_string())
    print(f"\nNegative delays (early): {(d < 0).sum():,}   > 120 min: {(d > 120).sum():,}")


def trips(df: pd.DataFrame) -> None:
    section("TRIPS")
    per_trip = df.groupby(["service_date", "trip_id"]).size()
    print(f"Trips (service_date, trip_id): {len(per_trip):,}")
    print(f"Stops per trip: min={per_trip.min()} median={per_trip.median():.0f} max={per_trip.max()}")
    dup = df.duplicated(["service_date", "trip_id", "stop_sequence"]).sum()
    print(f"Duplicate (service_date, trip_id, stop_sequence): {dup:,}")
    print("\nRows per service_date:")
    print(df["service_date"].value_counts().sort_index().to_string())


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    files = list_files(target)
    check_schemas(files)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    overview(df)
    categorical(df)
    delays(df)
    trips(df)


if __name__ == "__main__":
    main()
