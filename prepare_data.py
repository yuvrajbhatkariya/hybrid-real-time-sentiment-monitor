from kafka import KafkaProducer
import pandas as pd
import json
import time
from datetime import datetime

# Kafka producer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

df = pd.read_csv("/home/master/project/data/cleaned_reviews.csv")

df = df.head(2000).sample(frac=1).reset_index(drop=True)

print(f"Total Reviews: {len(df)}")

# Stream reviews
for i, row in df.iterrows():

    message = {
        'brand': str(row['brand']),
        'review_text': str(row['review_text']),
        'rating': int(row['rating']),
        'true_label': str(row['true_label']),
        'timestamp': datetime.now().isoformat()
    }

    try:

        producer.send(
            'reviews_stream',
            value=message
        )

        if i % 200 == 0:
            print(f"Sent {i} reviews")

    except Exception as e:
        print(f"Send error: {e}")

    # Slow streaming for stability
    time.sleep(2)

producer.flush()
producer.close()

print("Producer done.")