"""Create model features from pairs of consecutive stops.

Input:  data/clean/stop_pairs.parquet    (created by build_pairs.py)
        data/clean/clean_stops.parquet        (created by clean_data.py)
        data/clean/review_duplicates.parquet  (created by clean_data.py)
Output: data/clean/features.parquet      One row per pair: features + target

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
    # Trip history: how the delay of this train changed before it leaves stop N.
    "prev_hop_delay_change", "dwell_delay_change", "delay_change_3_stops",
    # Other trains in the last hour: on the same hop N -> N+1 and at the same stop N.
    "edge_recent_trains", "edge_recent_delay_change",
    "station_recent_trains", "station_recent_dwell_change",
    # Network load: how many trains the timetable plans at stop N in this hour.
    "station_trains_per_hour",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# The value the model will learn to predict.
TARGET = "target_next_arrival_delay_min"

# One stop of one trip: identifies every row of the pairs.
ROW_KEY = TRIP_COLUMNS + ["stop_sequence"]

# How far back to look at other trains, in minutes.
RECENT_MINUTES = 60

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


def add_trip_history_features(df):
    """Show how the delay of this train changed at the stops it has already passed.

    A train that is losing minutes now will probably keep losing them.
    Empty when an earlier stop or delay is missing: empty is not the same as "no change".
    """
    return (df
            # Minutes lost on the previous hop N-1 -> N (minus = time made up)
            .withColumn("prev_hop_delay_change",
                        F.col("arrival_delay_min") - F.col("prev_departure_delay_min"))
            # Minutes lost while standing at stop N
            .withColumn("dwell_delay_change",
                        F.col("departure_delay_min") - F.col("arrival_delay_min"))
            # Minutes lost over the last 3 hops, from leaving N-3 to leaving N
            .withColumn("delay_change_3_stops",
                        F.col("departure_delay_min") - F.col("back3_departure_delay_min")))


def real_time_seconds(planned_column, delay_column):
    """When something really happened: planned time + delay, in seconds."""
    return F.unix_timestamp(planned_column) + delay_column * 60


def recent_events(rows, events, place_columns, count_name, average_name):
    """Count and average events of OTHER trains at the same place in the last hour.

    rows:   ROW_KEY + place_columns + "now"           (seconds)
    events: TRIP_COLUMNS + place_columns + "known_at" (seconds) + "value"
    An event is used only if it happened before "now": the model may not see the future.
    """
    window = RECENT_MINUTES * 60
    # Compare only events from the same hour slot or the slot before,
    # so Spark does not compare every train with every other train.
    slot = F.floor(F.col("now") / window)
    rows = rows.withColumn("slot", F.explode(F.array(slot, slot - 1)))
    events = (events
              .withColumn("slot", F.floor(F.col("known_at") / window))
              .withColumnRenamed("service_date", "event_service_date")
              .withColumnRenamed("trip_id", "event_trip_id"))

    same_train = ((F.col("event_service_date") == F.col("service_date")) &
                  (F.col("event_trip_id") == F.col("trip_id")))
    recent = (rows.join(events, place_columns + ["slot"])
              .filter((F.col("known_at") >= F.col("now") - window) &
                      (F.col("known_at") < F.col("now")) &
                      ~same_train))
    return recent.groupBy(*ROW_KEY).agg(
        F.count("value").alias(count_name),
        F.avg("value").alias(average_name),
    )


def add_recent_traffic_features(df):
    """Describe what happened to other trains here in the last hour.

    There is no data about the causes of delays, but a problem on the line
    (a breakdown, works, a slow train ahead) shows up in the delays of other trains.
    """
    # "Now" is when our train really leaves stop N. If its departure delay
    # is unknown, use the arrival delay, or the timetable if both are empty.
    our_delay = F.coalesce(F.col("departure_delay_min"), F.col("arrival_delay_min"), F.lit(0.0))
    rows = df.select(*ROW_KEY, "station_id", "next_station_id",
                     real_time_seconds("scheduled_departure", our_delay).alias("now"))

    # Hop N -> N+1: other trains that already arrived at N+1,
    # and how many minutes they lost on this hop.
    hop_events = df.select(
        *TRIP_COLUMNS, "station_id", "next_station_id",
        real_time_seconds("next_scheduled_arrival", F.col(TARGET)).alias("known_at"),
        (F.col(TARGET) - F.col("departure_delay_min")).alias("value"),
    )
    # Stop N: other trains that already left N, and how many minutes they lost while standing there.
    stop_events = df.select(
        *TRIP_COLUMNS, "station_id",
        real_time_seconds("scheduled_departure", F.col("departure_delay_min")).alias("known_at"),
        (F.col("departure_delay_min") - F.col("arrival_delay_min")).alias("value"),
    )

    edge = recent_events(rows, hop_events.dropna(subset=["known_at", "value"]),
                         ["station_id", "next_station_id"],
                         "edge_recent_trains", "edge_recent_delay_change")
    station = recent_events(rows.drop("next_station_id"),
                            stop_events.dropna(subset=["known_at", "value"]),
                            ["station_id"],
                            "station_recent_trains", "station_recent_dwell_change")

    # No other trains means a count of 0. The average stays empty: there is nothing to average.
    return (df.join(edge, ROW_KEY, "left")
            .join(station, ROW_KEY, "left")
            .fillna(0, subset=["edge_recent_trains", "station_recent_trains"]))


def planned_hour(df):
    """The hour when a train is planned at a stop (departure, or arrival at the last stop)."""
    return F.date_trunc("hour", F.coalesce(df["scheduled_departure"], df["scheduled_arrival"]))


def add_network_load_features(df, stops):
    """Count how many different trains the timetable plans at stop N in the same hour.

    A busy station has more trains waiting for each other and for free platforms.
    The timetable is known in advance, so this does not reveal the target.
    Our own train is counted too, so the value is at least 1.
    """
    load = (stops
            .select(*TRIP_COLUMNS,
                    to_station_id("stop_id").alias("station_id"),
                    planned_hour(stops).alias("hour"))
            .dropna(subset=["station_id", "hour"])
            .groupBy("station_id", "hour")
            .agg(F.countDistinct(*TRIP_COLUMNS).alias("station_trains_per_hour")))
    return (df.withColumn("hour", planned_hour(df))
            .join(load, ["station_id", "hour"], "left")
            .drop("hour"))


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
        # but they still show which station the trip visits and when.
        # The planned times are needed to count trains per station and hour.
        columns = TRIP_COLUMNS + ["stop_sequence", "stop_id", "scheduled_arrival", "scheduled_departure"]
        stops = (spark.read.parquet(args.stops).select(*columns)
                 .unionByName(spark.read.parquet(args.duplicates).select(*columns)))
        pair_rows = pairs.count()

        # Add the first and last station to every pair.
        # "left" join keeps every pair, even if nothing is found.
        df = pairs.join(find_trip_ends(stops), TRIP_COLUMNS, "left")
        df = add_station_features(df)
        df = add_route_features(df)
        df = add_schedule_features(df)
        df = add_trip_history_features(df)
        df = add_recent_traffic_features(df)
        df = add_network_load_features(df, stops)

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
