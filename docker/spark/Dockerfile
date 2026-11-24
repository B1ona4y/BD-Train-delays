FROM apache/spark:4.0.1-python3

USER root
# Python libs needed on driver AND executors (MLlib needs numpy; Jupyter runs as the driver)
RUN pip install --no-cache-dir \
        numpy pandas pyarrow \
        jupyterlab matplotlib seaborn plotly \
        kafka-python
USER spark
