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

TRIP_COLUMNS = ["service_date", "trip_id"]
TARGET = "target_next_arrival_delay_min"


def build_pairs(stops):
    """Attach the following stop to each row and keep only true neighbours."""
    trip_order = Window.partitionBy(*TRIP_COLUMNS).orderBy("stop_sequence")

    def next_value(name):
        return F.lead(name).over(trip_order)

    # From stop N+1 take only what is known in advance, plus the target.
    joined = stops.select(
        *TRIP_COLUMNS,
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
        next_value("stop_sequence").alias("next_stop_sequence"),
        next_value("stop_id").alias("next_stop_id"),
        next_value("stop_name").alias("next_stop_name"),
        next_value("scheduled_arrival").alias("next_scheduled_arrival"),
        next_value("arrival_delay_min").alias(TARGET),
        # For evaluating results only. It reveals the target, so never use it as a feature.
        next_value("arrival_delay_outlier").alias("target_outlier"),
    )
    # A gap (for example 3 -> 5) means a stop was removed, so it is not a pair.
    return joined.filter(F.col("next_stop_sequence") == F.col("stop_sequence") + 1)


def main():
    parser = argparse.ArgumentParser(description="Build consecutive stop pairs using Spark")
    parser.add_argument("--input", default="/opt/project/data/clean/model_stops.parquet")
    parser.add_argument("--output", default="/opt/project/data/clean/stop_pairs.parquet")
    parser.add_argument("--master", default=os.environ.get("SPARK_MASTER", "local[*]"))
    args = parser.parse_args()

    spark = (SparkSession.builder.appName("RailPredict-Build-Pairs")
             .master(args.master)
             .config("spark.sql.session.timeZone", "UTC")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    try:
        stops = spark.read.parquet(args.input)
        stop_rows = stops.count()

        neighbours = build_pairs(stops)
        neighbour_rows = neighbours.count()

        pairs = neighbours.filter(F.col(TARGET).isNotNull() & ~F.isnan(TARGET))
        pairs.write.mode("overwrite").parquet(args.output)

        # Count what was actually saved.
        saved = spark.read.parquet(args.output)
        pair_rows = saved.count()

        print(f"\nStop records read:          {stop_rows:,}")
        print(f"Neighbouring pairs found:   {neighbour_rows:,}")
        print(f"Removed (no target):        {neighbour_rows - pair_rows:,}")
        print(f"Pairs saved:                {pair_rows:,}")
        print("\nMissing delays at stop N (kept, handled later in the Pipeline):")
        saved.select(*[F.sum(F.col(c).isNull().cast("long")).alias(c)
                       for c in ["arrival_delay_min", "departure_delay_min"]]).show()
        print("Pairs per day:")
        saved.groupBy("service_date").count().orderBy("service_date").show()
        print(f"Saved to: {args.output}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
