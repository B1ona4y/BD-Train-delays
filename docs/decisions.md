# Design decisions

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
