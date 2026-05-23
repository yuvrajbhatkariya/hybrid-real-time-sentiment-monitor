from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import os

# Create Spark session
spark = SparkSession.builder \
    .appName("SentimentStreaming") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Kafka schema
schema = StructType([
    StructField("brand", StringType()),
    StructField("review_text", StringType()),
    StructField("rating", IntegerType()),
    StructField("true_label", StringType()),
    StructField("timestamp", StringType())
])

# Read from Kafka
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "reviews_stream") \
    .load()

# Convert Kafka value
json_df = df.selectExpr(
    "CAST(value AS STRING)"
)

parsed_df = json_df.select(
    from_json(
        col("value"),
        schema
    ).alias("data")
).select("data.*")

# VADER analyzer
analyzer = SentimentIntensityAnalyzer()

# Sentiment prediction
def predict_sentiment(text):

    score = analyzer.polarity_scores(text)

    compound = score['compound']

    if compound >= 0.05:
        return "good"

    elif compound <= -0.05:
        return "poor"

    else:
        return "ok"

# UDF
predict_udf = udf(
    predict_sentiment,
    StringType()
)

final_df = parsed_df.withColumn(
    "predicted_label",
    predict_udf(col("review_text"))
)

OUTPUT_DIR = "/home/master/project/output"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# Drift memory
brand_history = {}

# Batch function
def process_batch(batch_df, batch_id):

    global brand_history

    pdf = batch_df.toPandas()

    if len(pdf) == 0:
        return

    print(f"\n=== Batch {batch_id} | Reviews: {len(pdf)} ===")

    # Sentiment counts
    counts = pdf['predicted_label'].value_counts().to_dict()

    print(counts)

    local_path = f"{OUTPUT_DIR}/batch_{batch_id}.csv"

    pdf.to_csv(
        local_path,
        index=False
    )

    # Save to HDFS
    try:

        hdfs_path = "/output/predictions"

        batch_df.write.mode("append").parquet(
            f"hdfs://master:9000{hdfs_path}"
        )

        print("HDFS write OK")

    except Exception as e:
        print(f"HDFS Write Error: {e}")

    # Drift detection
    drift_logs = []

    for brand in pdf['brand'].unique():

        brand_df = pdf[
            pdf['brand'] == brand
        ]

        poor_ratio = (
            (brand_df['predicted_label'] == 'poor').sum()
            / len(brand_df)
        )

        old_ratio = brand_history.get(
            brand,
            poor_ratio
        )

        if abs(poor_ratio - old_ratio) > 0.30:

            drift_logs.append({
                'brand': brand,
                'old_ratio': old_ratio,
                'new_ratio': poor_ratio,
                'drift_flag': 'DRIFT_ALERT'
            })

        brand_history[brand] = poor_ratio

    if drift_logs:

        drift_df = pd.DataFrame(drift_logs)

        drift_path = f"{OUTPUT_DIR}/drift_log.csv"

        if os.path.exists(drift_path):

            old = pd.read_csv(drift_path)

            drift_df = pd.concat([
                old,
                drift_df
            ])

        drift_df.to_csv(
            drift_path,
            index=False
        )

        print(
            f"Drift log saved: {len(drift_logs)} brands"
        )

# Streaming query
query = final_df.writeStream \
    .foreachBatch(process_batch) \
    .outputMode("append") \
    .option(
        "checkpointLocation",
        "/tmp/checkpoint_brand"
    ) \
    .start()

print("Streaming started. Waiting for Kafka data...")

query.awaitTermination()