import pandas as pd
from transformers import pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load batches
df = pd.read_csv(
    "/home/master/project/output/batch_1.csv"
)

# Load DistilBERT
classifier = pipeline(
    "sentiment-analysis"
)

def map_label(label):

    label = label.upper()

    if label == "POSITIVE":
        return "good"

    elif label == "NEGATIVE":
        return "poor"

    else:
        return "ok"

predictions = []

texts = df['review_text'].tolist()

for text in texts:

    result = classifier(text[:512])[0]

    pred = map_label(
        result['label']
    )

    predictions.append(pred)

df['distilbert_prediction'] = predictions

print("\nClassification Report:\n")

print(
    classification_report(
        df['true_label'],
        df['distilbert_prediction']
    )
)

cm = confusion_matrix(
    df['true_label'],
    df['distilbert_prediction']
)

fig, ax = plt.subplots(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['good', 'ok', 'poor'],
    yticklabels=['good', 'ok', 'poor']
)

plt.title("DistilBERT Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig(
    "/home/master/project/output/confusion_matrix_distilbert.png"
)

plt.show()

print("\nSaved:")
print("confusion_matrix_distilbert.png")