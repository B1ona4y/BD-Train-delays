"""Build pairs of consecutive stops: data about stop N -> arrival delay at stop N+1.

Input:  data/clean/model_stops.parquet   (created by clean_data.py)
Output: data/clean/stop_pairs.parquet    One row per pair of neighbouring stops

Run from the repository root:
  docker compose exec jupyter /opt/spark/bin/spark-submit --master spark://spark-master:7077 /opt/project/src/build_pairs.py
"""

import argparse
import os

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

# These two columns together identify one trip on one day.
TRIP_COLUMNS = ["service_date", "trip_id"]
# The value the model will learn to predict.
TARGET = "target_next_arrival_delay_min"


def build_pairs(stops):
    """Attach the following stop to each row and keep only true neighbours."""
    # Look at each trip separately, with its stops in the correct order.
    trip_order = Window.partitionBy(*TRIP_COLUMNS).orderBy("stop_sequence")

    # Take a value from the next row (the next stop) of the same trip.
    # For the last stop of a trip there is no next row, so the result is empty.
    def next_value(name):
        return F.lead(name).over(trip_order)

    # Put the current stop (N) and the next stop (N+1) into one row.
    # As model data, take from stop N+1 only what is known in advance, plus the target.
    joined = stops.select(
        *TRIP_COLUMNS,
        # Data about the trip and the current stop N.
        "route_id",
        "stop_sequence",
        "stop_id",
        "stop_name",
        "scheduled_arrival",
        "scheduled_departure",
        "arrival_delay_min",
        "departure_delay_min",
        "arrival_delay_outlier",
        "departure_delay_outlier",
        # Data about the next stop N+1.
        # Its number is needed to check that the two stops are neighbours.
        next_value("stop_sequence").alias("next_stop_sequence"),
        next_value("stop_id").alias("next_stop_id"),
        next_value("stop_name").alias("next_stop_name"),
        next_value("scheduled_arrival").alias("next_scheduled_arrival"),
        # The target: arrival delay at stop N+1, in minutes.
        next_value("arrival_delay_min").alias(TARGET),
        # For evaluating results only. It reveals the target, so never use it as a feature.
        next_value("arrival_delay_outlier").alias("target_outlier"),
    )
    # Keep a row only if the next number is the current number + 1.
    # A gap (for example 3 -> 5) means a stop is missing, so it is not a pair.
    # The last stop of each trip is also dropped here, because it has no next stop.
    return joined.filter(F.col("next_stop_sequence") == F.col("stop_sequence") + 1)


def main():
    # Settings that can be changed from the command line.
    parser = argparse.ArgumentParser(description="Build consecutive stop pairs using Spark")
    parser.add_argument("--input", default="/opt/project/data/clean/model_stops.parquet")
    parser.add_argument("--output", default="/opt/project/data/clean/stop_pairs.parquet")
    parser.add_argument("--master", default=os.environ.get("SPARK_MASTER", "local[*]"))
    args = parser.parse_args()

    # Connect to Spark. UTC keeps Spark from shifting times between time zones.
    spark = (SparkSession.builder.appName("RailPredict-Build-Pairs")
             .master(args.master)
             .config("spark.sql.session.timeZone", "UTC")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    try:
        # Read the cleaned stops (one row = one stop).
        stops = spark.read.parquet(args.input)
        stop_rows = stops.count()

        # Build the pairs (one row = two neighbouring stops).
        neighbours = build_pairs(stops)
        neighbour_rows = neighbours.count()

        # Remove pairs without a target: the model cannot learn from them.
        # Missing delays at the current stop N are kept.
        pairs = neighbours.filter(F.col(TARGET).isNotNull() & ~F.isnan(TARGET))
        # Save the result. "overwrite" replaces the previous one.
        pairs.write.mode("overwrite").parquet(args.output)

        # Read the saved data back and count what was actually written.
        saved = spark.read.parquet(args.output)
        pair_rows = saved.count()

        # Print a short report.
        print(f"\nStop records read:          {stop_rows:,}")
        print(f"Neighbouring pairs found:   {neighbour_rows:,}")
        print(f"Removed (no target):        {neighbour_rows - pair_rows:,}")
        print(f"Pairs saved:                {pair_rows:,}")
        # Count empty values in the current stop's delays.
        print("\nMissing delays at stop N (kept, handled later in the Pipeline):")
        saved.select(*[F.sum(F.col(c).isNull().cast("long")).alias(c)
                       for c in ["arrival_delay_min", "departure_delay_min"]]).show()
        # Show the number of pairs per day (useful for the train/test split).
        print("Pairs per day:")
        saved.groupBy("service_date").count().orderBy("service_date").show()
        print(f"Saved to: {args.output}")
    # Always close Spark, even if an error happened above.
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
