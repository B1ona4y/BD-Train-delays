"""Create model features from pairs of consecutive stops.

Input:  data/clean/stop_pairs.parquet    (created by build_pairs.py)
        data/clean/clean_stops.parquet        (created by clean_data.py)
        data/clean/review_duplicates.parquet  (created by clean_data.py)
Output: data/clean/features.parquet      One row per pair: 14 features + target

Text features stay as text and missing numbers stay empty.
Both will be handled later in the MLlib Pipeline.

Run from the repository root:
  docker compose exec jupyter /opt/spark/bin/spark-submit --master spark://spark-master:7077 /opt/project/src/build_features.py
"""

import argparse
import os

from pyspark.sql import SparkSession, functions as F

TRIP_COLUMNS = ["service_date", "trip_id"]

# Groups (names). The Pipeline will turn them into numbers.
CATEGORICAL_FEATURES = [
    "route_base", "carrier", "first_station_id", "last_station_id",
    "station_id", "next_station_id",
]
# Values that are already numbers.
NUMERIC_FEATURES = [
    "is_bus", "stop_sequence", "departure_hour", "day_of_week",
    "scheduled_travel_min", "scheduled_dwell_min",
    "arrival_delay_min", "departure_delay_min",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# The value the model will learn to predict.
TARGET = "target_next_arrival_delay_min"

# These columns are saved too, but they are NOT features.
# service_date: train/test split. trip_id: checks. target_outlier: evaluating results.
EXTRA_COLUMNS = TRIP_COLUMNS + ["target_outlier"]


def to_station_id(stop_id_column):
    """Keep only the station number: 37796_RAIL_1_1 -> 37796."""
    # The part after the first "_" is ignored, so all IDs of one station match.
    # An empty stop_id stays empty.
    return F.split(F.col(stop_id_column), "_").getItem(0)


def add_station_features(df):
    """Create station numbers for the current stop (N) and the next stop (N+1)."""
    return (df
            .withColumn("station_id", to_station_id("stop_id"))
            .withColumn("next_station_id", to_station_id("next_stop_id")))


def find_trip_ends(stops):
    """Find the first and last station of every trip."""
    # Compare (stop number, stop_id) together: the smallest pair is the first stop,
    # the largest is the last one. If a stop number appears twice, the result
    # is still the same on every run.
    stop = F.struct("stop_sequence", "stop_id")
    ends = stops.groupBy(*TRIP_COLUMNS).agg(
        F.min(stop).getField("stop_id").alias("first_stop_id"),
        F.max(stop).getField("stop_id").alias("last_stop_id"),
    )
    return ends.select(
        *TRIP_COLUMNS,
        to_station_id("first_stop_id").alias("first_station_id"),
        to_station_id("last_stop_id").alias("last_station_id"),
    )


def add_route_features(df):
    """Split route_id into parts the model can use."""
    # Use "unknown" when the route is empty.
    route = F.coalesce(F.col("route_id"), F.lit("unknown"))
    # Remove "_BUS" from the end: KM_R2_BUS -> KM_R2
    route_base = F.regexp_replace(route, "_BUS$", "")

    return (df
            .withColumn("route_base", route_base)
            # The first part of the name: KM_R2 -> KM
            .withColumn("carrier", F.split(route_base, "_").getItem(0))
            # 1 if the name ends with "_BUS", otherwise 0
            .withColumn("is_bus", route.endswith("_BUS").cast("int")))


def minutes_between(start, end):
    """Number of minutes from one time column to another."""
    return (F.unix_timestamp(end) - F.unix_timestamp(start)) / 60.0


def add_schedule_features(df):
    """Create features from the timetable (known before the trip)."""
    return (df
            # Hour of the scheduled departure from stop N (0-23)
            .withColumn("departure_hour", F.hour("scheduled_departure"))
            # Spark numbering: 1 = Sunday, 2 = Monday, ..., 7 = Saturday
            .withColumn("day_of_week", F.dayofweek("service_date"))
            # Planned time from stop N to stop N+1
            .withColumn("scheduled_travel_min",
                        minutes_between("scheduled_departure", "next_scheduled_arrival"))
            # Planned waiting time at stop N
            .withColumn("scheduled_dwell_min",
                        minutes_between("scheduled_arrival", "scheduled_departure")))


def main():
    # Settings that can be changed from the command line.
    parser = argparse.ArgumentParser(description="Create model features using Spark")
    parser.add_argument("--pairs", default="/opt/project/data/clean/stop_pairs.parquet")
    parser.add_argument("--stops", default="/opt/project/data/clean/clean_stops.parquet")
    parser.add_argument("--duplicates", default="/opt/project/data/clean/review_duplicates.parquet")
    parser.add_argument("--output", default="/opt/project/data/clean/features.parquet")
    parser.add_argument("--master", default=os.environ.get("SPARK_MASTER", "local[*]"))
    args = parser.parse_args()
    # "overwrite" deletes the output first, so it must never be an input.
    inputs = [args.pairs, args.stops, args.duplicates]
    if os.path.abspath(args.output) in map(os.path.abspath, inputs):
        parser.error("--output must be different from the input paths")

    # Connect to Spark. Use the same time zone as the earlier scripts.
    spark = (SparkSession.builder.appName("RailPredict-Build-Features")
             .master(args.master)
             .config("spark.sql.session.timeZone", "UTC")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    try:
        pairs = spark.read.parquet(args.pairs)
        # For the first and last station we need the full route of a trip:
        # model_stops has some stops removed, and clean_stops does not include
        # conflicting duplicates. Their records differ in times and delays,
        # but they still show which station the trip visits.
        columns = TRIP_COLUMNS + ["stop_sequence", "stop_id"]
        stops = (spark.read.parquet(args.stops).select(*columns)
                 .unionByName(spark.read.parquet(args.duplicates).select(*columns)))
        pair_rows = pairs.count()

        # Add the first and last station to every pair.
        # "left" join keeps every pair, even if nothing is found.
        df = pairs.join(find_trip_ends(stops), TRIP_COLUMNS, "left")
        df = add_station_features(df)
        df = add_route_features(df)
        df = add_schedule_features(df)

        # Keep only the features, the target and the extra columns.
        result = df.select(*EXTRA_COLUMNS, *FEATURES, TARGET)
        # Save the result. "overwrite" replaces the previous one.
        result.write.mode("overwrite").parquet(args.output)

        # Read the saved data back and check it.
        saved = spark.read.parquet(args.output)
        saved_rows = saved.count()
        # This step only adds columns, so the number of rows must not change.
        if saved_rows != pair_rows:
            raise RuntimeError(f"Row count changed: {pair_rows} -> {saved_rows}")

        print(f"\nRows saved:     {saved_rows:,}")
        print(f"Features:       {len(FEATURES)}")

        print("\nEmpty values per feature:")
        empty = saved.select(*[F.sum(F.col(c).isNull().cast("long")).alias(c)
                               for c in FEATURES]).first().asDict()
        for name, count in empty.items():
            print(f"  {name:24}{count:>9,}")

        print("\nRanges of numeric features:")
        saved.select(*NUMERIC_FEATURES).summary("min", "50%", "max").show(vertical=True)

        print("Different values per categorical feature:")
        saved.select(*[F.countDistinct(c).alias(c) for c in CATEGORICAL_FEATURES]).show(vertical=True)

        # Negative planned times would mean a problem in the timetable data.
        for name in ["scheduled_travel_min", "scheduled_dwell_min"]:
            print(f"Negative {name}: {saved.filter(F.col(name) < 0).count():,}")

        print(f"\nSaved to: {args.output}")
    # Always close Spark, even if an error happened above.
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
