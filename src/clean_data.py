"""Clean the raw railway data (one row = one stop of one trip).

Input:  data/dirty/*.parquet

Outputs (all within data/clean/):
  clean_stops.parquet/           Unique, validated stop records (all statuses)
  model_stops.parquet/           Stops eligible for ML feature engineering
  review_duplicates.parquet/     Conflicting versions of the same stop
  invalid_keys.parquet/         Rows lacking a usable stop identifier
  quality_report.json           Reproducible row counts and quality statistics
  model_filter_report.json      Counts of excluded rows and flagged outliers

Run from the repository root:
  docker compose exec jupyter /opt/spark/bin/spark-submit --master spark://spark-master:7077 /opt/project/src/clean_data.py
"""

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StringType

LOG = logging.getLogger("railpredict.etl")

REQUIRED_COLUMNS = [
    "service_date", "trip_id", "stop_sequence", "trip_status", "stop_status",
    "arrival_delay_min", "departure_delay_min", "scheduled_arrival",
    "scheduled_departure", "predicted_arrival", "predicted_departure",
    "observed_at", "delay_source",
]
# These three columns together identify one stop of one trip.
KEY_COLUMNS = ["service_date", "trip_id", "stop_sequence"]
# Columns converted from text to date and time.
TIMESTAMP_COLUMNS = [
    "scheduled_arrival", "scheduled_departure", "predicted_arrival",
    "predicted_departure", "observed_at",
]
DELAY_COLUMNS = ["arrival_delay_min", "departure_delay_min"]
# These columns are removed only if they contain no values at all.
DROP_ALWAYS_EMPTY = ["feed_id", "trip_start_time"]


def aggregate_counts(df):
    """Count rows, missing delays, cancelled and skipped records."""
    return df.agg(
        F.count("*").alias("rows"),
        F.sum(F.col("arrival_delay_min").isNull().cast("long")).alias("missing_arrival_delay"),
        F.sum(F.col("departure_delay_min").isNull().cast("long")).alias("missing_departure_delay"),
        F.sum((F.col("trip_status") == "cancelled").cast("long")).alias("cancelled"),
        F.sum((F.col("stop_status") == "skipped").cast("long")).alias("skipped"),
    ).first().asDict()


def main():
    # Settings that can be changed from the command line.
    parser = argparse.ArgumentParser(description="Clean railway stop data using Spark")
    parser.add_argument("--input", default="/opt/project/data/dirty")
    parser.add_argument("--output", default="/opt/project/data/clean")
    parser.add_argument("--master", default=os.environ.get("SPARK_MASTER", "local[*]"))
    parser.add_argument("--outlier-threshold", type=float, default=60.0,
                        help="Flag absolute delays above this number of minutes (do not remove)")
    parser.add_argument("--include-propagated", action="store_true",
                        help="Keep propagated delay records in model_stops.parquet (excluded by default)")
    args = parser.parse_args()
    if args.outlier_threshold <= 0:
        parser.error("--outlier-threshold must be greater than zero")

    # Set up log messages and connect to Spark.
    # UTC keeps Spark from shifting times between time zones.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    spark = (SparkSession.builder.appName("RailPredict-Clean-Stops")
             .master(args.master)
             .config("spark.sql.session.timeZone", "UTC")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    try:
        # 1. Read the raw data and stop early if something important is missing.
        raw = spark.read.parquet(args.input)
        missing = sorted(set(REQUIRED_COLUMNS) - set(raw.columns))
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        raw_rows = raw.count()
        LOG.info("Input records: %s", raw_rows)
        if raw_rows == 0:
            raise ValueError("Input dataset is empty")

        # 2. Normalize text and data types.
        # Normalize whitespace and blank strings before parsing timestamps.
        df = raw
        for field in df.schema.fields:
            if isinstance(field.dataType, StringType):
                name = field.name
                df = df.withColumn(name, F.when(F.trim(F.col(name)) == "", F.lit(None))
                                   .otherwise(F.trim(F.col(name))))

        # Data type normalization. Invalid values become NULL and are measurable.
        df = df.withColumn("service_date", F.to_date("service_date", "yyyy-MM-dd"))
        df = df.withColumn("stop_sequence", F.col("stop_sequence").cast("long"))
        for name in DELAY_COLUMNS:
            df = df.withColumn(name, F.col(name).cast("double"))
        for name in TIMESTAMP_COLUMNS:
            df = df.withColumn(name, F.to_timestamp(name, "yyyy-MM-dd HH:mm:ss"))

        # 3. Remove columns that are completely empty.
        removed_empty_columns = [name for name in DROP_ALWAYS_EMPTY
                                 if name in df.columns and df.filter(F.col(name).isNotNull()).limit(1).count() == 0]
        df = df.drop(*removed_empty_columns)

        # 4. Set aside rows that cannot be identified.
        invalid_key = (
                F.col("service_date").isNull() |
                F.col("trip_id").isNull() |
                F.col("stop_sequence").isNull() |
                (F.col("stop_sequence") < 0)
        )
        invalid = df.filter(invalid_key)
        with_keys = df.filter(~invalid_key)
        invalid_rows = invalid.count()

        # 5. Remove duplicates.
        # Remove exact duplicates only
        equality_columns = [name for name in with_keys.columns if name != "source_file"]
        exact_deduped = with_keys.dropDuplicates(equality_columns)
        exact_duplicates_removed = with_keys.count() - exact_deduped.count()

        # Never choose a random record when versions of a stop conflict.
        multiplicities = (exact_deduped.groupBy(*KEY_COLUMNS)
                          .count().filter(F.col("count") > 1)
                          .select(*KEY_COLUMNS))
        conflicting_key_count = multiplicities.count()
        conflicting = exact_deduped.join(multiplicities, KEY_COLUMNS, "inner")
        clean = exact_deduped.join(multiplicities, KEY_COLUMNS, "left_anti")

        # Check that every input row is accounted for.
        clean_rows = clean.count()
        conflicting_rows = conflicting.count()
        assert raw_rows == (invalid_rows + exact_duplicates_removed + conflicting_rows + clean_rows), (
            "Row accounting mismatch: check ETL transformations"
        )

        # Statistics for the quality report.
        stats = aggregate_counts(clean)
        null_counts = clean.agg(*[
            F.sum(F.col(c).isNull().cast("long")).alias(c)
            for c in clean.columns
        ]).first().asDict()

        # 6. Select the stops that can be used for the model.
        # Keep normal trips only.
        scheduled = clean.filter(F.col("trip_status") == "scheduled")
        excluded_trip_status = clean_rows - scheduled.count()

        # Remove stops the train did not serve. An empty stop_status is kept.
        non_skipped = scheduled.filter(
            F.col("stop_status").isNull() | (F.col("stop_status") != "skipped")
        )
        excluded_skipped = scheduled.count() - non_skipped.count()

        # By default keep only delays whose source is "feed".
        if args.include_propagated:
            model_base = non_skipped
            excluded_delay_source = 0
        else:
            model_base = non_skipped.filter(F.col("delay_source") == "feed")
            excluded_delay_source = non_skipped.count() - model_base.count()

        # 7. Mark very large delays. They are NOT removed.
        threshold = args.outlier_threshold
        model_stops = (model_base
                       .withColumn("arrival_delay_outlier",
                                   F.coalesce(F.abs(F.col("arrival_delay_min")) > F.lit(threshold), F.lit(False)))
                       .withColumn("departure_delay_outlier",
                                   F.coalesce(F.abs(F.col("departure_delay_min")) > F.lit(threshold), F.lit(False))))

        model_stats = model_stops.agg(
            F.count("*").alias("rows"),
            F.sum(F.col("arrival_delay_outlier").cast("long")).alias("arrival_outlier_rows"),
            F.sum(F.col("departure_delay_outlier").cast("long")).alias("departure_outlier_rows"),
            F.sum(F.col("arrival_delay_min").isNull().cast("long")).alias("missing_arrival_delay"),
            F.sum(F.col("departure_delay_min").isNull().cast("long")).alias("missing_departure_delay"),
        ).first().asDict()
        model_rows = model_stats["rows"]
        assert clean_rows == (excluded_trip_status + excluded_skipped + excluded_delay_source + model_rows), (
            "Model filter row accounting mismatch"
        )

        # 8. Save the datasets. "overwrite" replaces previous results.
        output = args.output.rstrip("/")
        clean.write.mode("overwrite").parquet(f"{output}/clean_stops.parquet")
        model_stops.write.mode("overwrite").parquet(f"{output}/model_stops.parquet")
        conflicting.write.mode("overwrite").parquet(f"{output}/review_duplicates.parquet")
        invalid.write.mode("overwrite").parquet(f"{output}/invalid_keys.parquet")

        # 9. Save two reports: what was cleaned and what was filtered.
        report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_path": args.input,
            "output_path": args.output,
            "original_rows": raw_rows,
            "exact_duplicate_rows_removed": exact_duplicates_removed,
            "invalid_key_rows_quarantined": invalid_rows,
            "conflicting_keys": conflicting_key_count,
            "conflicting_rows_quarantined": conflicting_rows,
            "clean_rows": clean_rows,
            "clean_statistics": stats,
            "missing_by_column_after_cleaning": null_counts,
            "all_null_columns_removed": removed_empty_columns,
            "rules": [
                "Trim whitespace and normalize empty strings to null",
                "Parse date/time columns and cast delay columns to double",
                "Quarantine missing or invalid logical keys",
                "Remove identical records, preserving one copy",
                "Quarantine all conflicting records sharing the same logical key",
                "Keep missing delay values, cancellations and skipped stops in clean_stops for analysis",
                "Preserve negative and extreme delays; do not clip or impute them",
                "Do not treat predicted arrival times as verified actual arrivals",
            ],
        }

        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        report_path = output_path / "quality_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
                               encoding="utf-8")

        model_report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_dataset": f"{output}/clean_stops.parquet",
            "output_dataset": f"{output}/model_stops.parquet",
            "starting_clean_rows": clean_rows,
            "excluded_trip_status_not_scheduled": excluded_trip_status,
            "excluded_skipped_stops": excluded_skipped,
            "excluded_non_feed_delay_source": excluded_delay_source,
            "include_propagated": args.include_propagated,
            "outlier_threshold_abs_minutes": threshold,
            "outliers_are_flagged_not_removed": True,
            "model_statistics": model_stats,
            "notes": [
                "model_stops is an intermediate dataset, not final training pairs",
                "Missing delay values remain until the next-stop label is constructed",
                "Source-reported delays may be predictions rather than verified actual arrival times",
                "Outliers are flagged but retained",
            ],
        }
        model_report_path = output_path / "model_filter_report.json"
        model_report_path.write_text(json.dumps(model_report, indent=2, ensure_ascii=False, default=str) + "\n",
                                     encoding="utf-8")

        # 10. Print a short summary.
        LOG.info("Clean records: %s", clean_rows)
        LOG.info("Model stop records: %s", model_rows)
        LOG.info("Model exclusions - train status: %s; skipped stop: %s; non-feed delay source: %s",
                 excluded_trip_status, excluded_skipped, excluded_delay_source)
        LOG.info("Outliers flagged - arrival: %s; departure: %s; threshold: +/-%.1f min",
                 model_stats["arrival_outlier_rows"], model_stats["departure_outlier_rows"], threshold)
        LOG.info("Model report: %s", model_report_path)
        LOG.info("Exact duplicate rows removed: %s", exact_duplicates_removed)
        LOG.info("Invalid-key records quarantined: %s", invalid_rows)
        LOG.info("Conflicting keys: %s; records quarantined: %s", conflicting_key_count, conflicting_rows)
        LOG.info("Report: %s", report_path)
        LOG.info("Finished ETL successfully")
    # Always close Spark, even if an error happened above.
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
