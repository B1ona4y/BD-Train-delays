"""Check the quality of the raw data. It only prints results and changes nothing.

Run from the repository root:
  docker compose exec jupyter /opt/spark/bin/spark-submit --master spark://spark-master:7077 /opt/project/src/inspect_quality.py
"""

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

spark = (
    SparkSession.builder
    .appName("railpredict-quality-check")
    .master("spark://spark-master:7077")
    .getOrCreate()
)

df = spark.read.parquet("/opt/project/data/dirty")

# These three columns together identify one stop of one trip.
keys = ["service_date", "trip_id", "stop_sequence"]

print("\n1. DUPLICATE KEYS")

# Stops that appear more than once, and whether their delays differ.
duplicates = (
    df.groupBy(*keys)
    .agg(
        F.count("*").alias("rows"),
        F.countDistinct(
            "arrival_delay_min"
        ).alias("different_arrival_delays")
    )
    .filter(F.col("rows") > 1)
)

duplicates.groupBy(
    "rows", "different_arrival_delays"
).count().show(30)

print("\n2. DELAY PREDICTION CONSISTENCY")

# Is the delay equal to (predicted arrival - scheduled arrival)?
check = (
    df.withColumn(
        "scheduled_ts",
        F.to_timestamp("scheduled_arrival")
    )
    .withColumn(
        "predicted_ts",
        F.to_timestamp("predicted_arrival")
    )
    .withColumn(
        "calculated_delay",
        (
            F.unix_timestamp("predicted_ts")
            - F.unix_timestamp("scheduled_ts")
        ) / 60.0
    )
    .filter(
        F.col("arrival_delay_min").isNotNull()
        & F.col("calculated_delay").isNotNull()
    )
)

check.select(
    F.count("*").alias("compared_rows"),
    F.avg(
        (F.abs(
            F.col("arrival_delay_min")
            - F.col("calculated_delay")
        ) < 0.11).cast("double")
    ).alias("matching_fraction")
).show()

print("\n3. STATUSES AND MISSING DELAY VALUES")

# How often delays are missing for each kind of record.
df.groupBy(
    "trip_status", "stop_status", "delay_source"
).agg(
    F.count("*").alias("rows"),
    F.sum(
        F.col("arrival_delay_min").isNull().cast("int")
    ).alias("missing_arrivals")
).orderBy(F.desc("rows")).show(30, truncate=False)

print("\n4. CONSECUTIVE STATION PAIRS")

# How many pairs of neighbouring stops can be built from non-duplicated rows.
w = Window.partitionBy(
    "service_date", "trip_id"
).orderBy("stop_sequence")

unique_keys = (
    df.groupBy(*keys).count()
    .filter(F.col("count") == 1)
    .select(*keys)
)

single = df.join(unique_keys, keys, "inner")

pairs = (
    single.withColumn(
        "next_sequence", F.lead("stop_sequence").over(w)
    )
    .withColumn(
        "next_arrival_delay",
        F.lead("arrival_delay_min").over(w)
    )
    .filter(
        (F.col("next_sequence") == F.col("stop_sequence") + 1)
        & F.col("departure_delay_min").isNotNull()
        & F.col("next_arrival_delay").isNotNull()
        & (F.col("trip_status") == "scheduled")
    )
)

print("Pairs Requiring Further Verification:", pairs.count())

spark.stop()
