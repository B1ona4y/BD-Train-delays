# BD-Train-delays
Big Data project: predicting delays of Polish trains (all operators) with Apache Spark MLlib and Kafka.

## How to run

You need Docker with Docker Compose. Run all commands from the repository root.
The raw data (17-23 Nov 2026) is already in `data/dirty/`.

**1. Start the containers**
```bash
docker compose up -d --build
```

**2. Allow the containers to write files** (first run only, Linux)
```bash
mkdir -p data/clean
chmod 777 data/clean notebooks
chmod 666 notebooks/model.ipynb
```

**3. Clean the raw data**
```bash
docker compose exec jupyter /opt/spark/bin/spark-submit --master spark://spark-master:7077 /opt/project/src/clean_data.py
```

**4. Build pairs of neighbouring stops**
```bash
docker compose exec jupyter /opt/spark/bin/spark-submit --master spark://spark-master:7077 /opt/project/src/build_pairs.py
```

**5. Create the features**
```bash
docker compose exec jupyter /opt/spark/bin/spark-submit --master spark://spark-master:7077 /opt/project/src/build_features.py
```

**6. Split into train and test:** open http://localhost:8888, then `notebooks/model.ipynb`, and run all cells.
Afterwards choose **Kernel → Shut Down Kernel**, otherwise the next Spark job may wait for free cores.

Results are saved in `data/clean/`. If you rerun a step, rerun all the steps after it.
To stop the containers: `docker compose down`.

## To be continued...

The data is ready: one row is one hop between two stations, and the goal is the arrival delay at the next one.

But there is a rival waiting. The laziest forecast possible, *"the delay will not change"*,
is already within 1 minute for about 95% of hops.

**Can an MLlib model beat it? Find out in the pipeline branch...** 🚆
