"""Read the raw Parquet data with Spark and print its structure.

Also serves as the check "Spark can read the data".

Run inside the cluster (recommended):
    docker compose exec jupyter spark-submit --master spark://spark-master:7077 \
        /opt/project/src/spark_inspect.py

Or locally (needs Java 17/21):
    python src/spark_inspect.py --master "local[*]"
"""

import argparse
import os
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "dirty"
DELAYS = ["arrival_delay_min", "departure_delay_min"]
CATEGORICAL = ["provider", "route_id", "trip_status", "stop_status", "delay_source"]


def section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--path", default=str(DATA_DIR))
    p.add_argument("--master", default=os.getenv("SPARK_MASTER", "local[*]"))
    args = p.parse_args()

    spark = SparkSession.builder.appName("inspect-raw-data").master(args.master).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # files are named date=YYYY-MM-DD.parquet (files, not partition dirs) -> keep the file name
    df = spark.read.parquet(args.path).withColumn("source_file", F.element_at(F.split(F.input_file_name(), "/"), -1))
    df.cache()

    section("SCHEMA")
    df.printSchema()

    total = df.count()
    section(f"ROWS: {total:,}   PARTITIONS: {df.rdd.getNumPartitions()}")
    df.groupBy("source_file").count().orderBy("source_file").show(truncate=False)

    section("SAMPLE")
    df.show(5, truncate=False, vertical=True)

    section("NULLS PER COLUMN")
    nulls = df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]).first().asDict()
    for col, n in nulls.items():
        print(f"{col:22} {n:>9,}  ({n / total:6.1%})")

    section("DISTINCT VALUES")
    df.select([F.countDistinct(c).alias(c) for c in df.columns if c != "source_file"]).show(vertical=True)

    section("CATEGORICAL VALUES")
    for c in CATEGORICAL:
        df.groupBy(c).count().orderBy(F.desc("count")).show(15, truncate=False)

    section("DELAYS (minutes)")
    df.select(DELAYS).summary("count", "mean", "stddev", "min", "1%", "25%", "50%", "75%", "99%", "max").show()

    section("TARGET CLASSES (arrival_delay_min)")
    d = F.col("arrival_delay_min")
    (df.withColumn("delay_class", F.when(d.isNull(), "null").when(d <= 3, "0-3").when(d <= 10, "4-10").otherwise(">10"))
       .groupBy("delay_class").count()
       .withColumn("pct", F.round(F.col("count") / total * 100, 1))
       .orderBy("delay_class").show())

    section("TRIPS")
    per_trip = df.groupBy("service_date", "trip_id").count()
    per_trip.select(F.count("*").alias("trips"), F.min("count").alias("min_stops"),
                    F.expr("percentile_approx(count, 0.5)").alias("median_stops"),
                    F.max("count").alias("max_stops")).show()
    dups = total - df.dropDuplicates(["service_date", "trip_id", "stop_sequence"]).count()
    print(f"Duplicate (service_date, trip_id, stop_sequence): {dups:,}")

    spark.stop()


if __name__ == "__main__":
    main()
