# Design decisions

1. [Broadcast joins are disabled in `build_features.py`](#1-broadcast-joins-are-disabled-in-build_featurespy)
2. [Train and test are split by date, not randomly](#2-train-and-test-are-split-by-date-not-randomly)

Known limitations of the data are listed at the end, in [Known limitations](#known-limitations).

## 1. Broadcast joins are disabled in `build_features.py`

**Decision.** `build_features.py` starts Spark with
`spark.sql.autoBroadcastJoinThreshold = -1`, so Spark never uses broadcast joins in this script.

### Problem
After the data grew from 7 days (17–23 Nov, 769,950 pairs) to 10 days
(17–26 Nov, 1,115,191 pairs), step 5 failed while writing `features.parquet`:

```
SparkException: Not enough memory to build and broadcast the table to all worker nodes.
```

### Why it happened
Spark has two main ways to join two tables:

| Join | How it works | When Spark picks it |
|---|---|---|
| Broadcast join | The driver collects the smaller table and sends a full copy to every worker. No shuffle is needed. | The table looks smaller than `autoBroadcastJoinThreshold` (10 MB by default) |
| Sort-merge join | Both tables are split by the join key, sorted and merged on the workers. | Otherwise |

A broadcast join is fast, but the whole table must fit into the **driver's** memory.
In this project the driver has the default 1 GB.

The new features (recent traffic, network load and edge history) join the pairs with
tables built from the pairs themselves or from all stops. With more data, one of these tables became too large to
broadcast, but Spark still chose a broadcast join. Its decision is based on a size
estimate, not on free memory. We did not check which of the joins caused the failure.

### Options we considered
| Option | Why we did not choose it / why we chose it |
|---|---|
| Give the driver more memory (`spark.driver.memory`) | Only moves the limit: it fails again when more days are added. It also needs changes to the Docker setup. |
| Raise or lower the threshold | Depends on the data size, so it can break again for the same reason. |
| **Disable broadcast joins (chosen)** | Every join becomes a sort-merge join. It runs on the workers and can use disk when memory is full, so it works for any data size. |

### Cost
Sort-merge joins need a shuffle, so for small data they can be slower than broadcast joins.
In our case step 5 takes about 1.5 minutes for 1.1 million pairs, which is acceptable.

### How it was checked
- Step 5 finished on the real data: 1,115,191 rows in and out, 25 features.
- A second run into a temporary folder gave exactly the same table
  (compared row by row with pandas).
- The setting only changes **how** Spark joins, not **what** it joins,
  so the features are the same as with broadcast joins.

## 2. Train and test are split by date, not randomly

**Decision.** `notebooks/model.ipynb` splits the pairs by `service_date`:

| Part | Dates | Used for |
|---|---|---|
| fit | 17–22 Nov | training model variants |
| validation | 23 Nov | comparing the variants |
| test | 24–26 Nov | the final check only |

`build_features.py --stats-end` is set to 22 Nov, the last fit day (see below).

### Why not a random split
1. **Neighbouring rows of one trip are almost the same.** One row is one hop of one train.
   The current delay in row N+1 is exactly the target of row N. With a random split,
   row N+1 can be in train and row N in test, so the test checks memory, not prediction.
2. **Rows of one day share events.** A breakdown, works or bad weather affects many trains
   on the same day. With a random split the model would already "know" that day in the test.
3. **Some features look only into the past.** The recent traffic features use other trains of the
   last hour, and the edge history uses whole earlier days. With a random split, targets of test rows
   would end up in features of training rows.
4. **The real task is a forecast.** A model is trained on past days and used on the next day.
   A split by date checks exactly this.

Dropping `day_of_week` would not help: the rows stay connected through the trip and the day.

### Why these dates
After the data grew to 10 days (17–26 Nov), the old dates (test 22–23 Nov) would have put
all new days into the test, so it would be half of the data. The new dates keep about
60% / 10% / 30% of the rows for fit / validation / test.

### What `--stats-end` has to do with it
The edge history features (`edge_typical_delay_change`, `edge_share_gain_3min`, `edge_history_hops`)
are learned from days up to `--stats-end`. It must be the last **fit** day: then neither the
validation day nor the test days are used to build features. A row also never uses its own day.

### Limitation
17 Nov 2026 is a Tuesday, so each day of the week appears only once or twice.
The validation day (23 Nov) is the only Monday, and validation and test have no weekend days.
Features that depend on the day of the week (`day_of_week`) cannot be checked well with this data.

## Known limitations

### L1. Wrong last station for some international trains

**What is wrong.** Some international PKP Intercity trains (between Poland and Germany or Czechia)
end with an extra row that is not a real stop. Example: a train from Berlin to Warsaw, 17 Nov:

| stop_sequence | stop_name | Planned time | Delay |
|---|---|---|---|
| 9 | Warszawa Centralna | 13:00 | −3 |
| 10 | **Warszawa Wschodnia** (real last stop) | 13:15 | −2 |
| 1000 | Frankfurt (Oder) | none | none |

The extra row has no planned times and no delays, its `stop_sequence` is 1000 or more,
and its `stop_id` ends with `_FALLBACK`. It looks like a technical record of the data source.
In the data for 17–26 Nov there are 17 such rows: 7 trains, 17 train runs.

**What it affects.** `build_features.py` takes the stop with the highest number as the last one, so for these runs:
- `last_station_id` points to the extra row (for example Frankfurt instead of Warsaw);
- `to_big_city` can be 0 although the train goes to Warsaw;
- `stops_to_end` is one stop too large.

**Why it is not fixed yet.** Only 9 of 1,115,191 feature rows belong to these runs
(most of their stops do not form pairs), so the effect on the model is negligible.

**How to fix it later.** In `clean_data.py`, set aside rows that have neither a planned arrival
nor a planned departure, because such a row is not a real stop. Before doing this, check the other
194 rows without planned times: they have no `stop_id` and no `stop_name`, so they are a different case.
