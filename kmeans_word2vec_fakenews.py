# Import necessary libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
import numpy as np
import graphviz
from sklearn.cluster import KMeans

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load data and drop rows with NaN values
data = pd.read_csv('WELFake_Dataset.csv')
data.dropna(subset=['text', 'label'], inplace=True)
print("Data loaded and cleaned. Remaining NaN values:")
print(data.isnull().sum())

# Text preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        words = [word for word in text.split() if word not in stop_words]
        return words
    else:
        return []

# Preprocess the data
print("Starting data preprocessing...")
data['cleaned_text'] = data['text'].apply(preprocess_text)
print("Data preprocessing completed.")

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data['cleaned_text'], data['label'], test_size=0.2, random_state=42
)

# Train the Word2Vec model
print("Training Word2Vec model...")
w2v_model = Word2Vec(sentences=X_train, vector_size=100, window=5, min_count=1, workers=4)
print("Word2Vec model training completed.")

# Convert text to the average Word2Vec vector
def text_to_vector(text):
    vectors = [w2v_model.wv[word] for word in text if word in w2v_model.wv]
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(w2v_model.vector_size)

X_train_vect = np.array([text_to_vector(text) for text in X_train])
X_test_vect = np.array([text_to_vector(text) for text in X_test])

# Train the KMeans clustering model
print("Training KMeans clustering model...")
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train_vect)
print("Model training completed.")

# Predict cluster labels on the test set
y_pred_clusters = kmeans.predict(X_test_vect)

from collections import Counter

# Map cluster labels to true labels
def map_clusters_to_labels(y_true, y_clusters):
    labels = np.unique(y_clusters)
    label_mapping = {}
    for label in labels:
        indices = np.where(y_clusters == label)
        true_labels_in_cluster = y_true[indices]
        if len(true_labels_in_cluster) == 0:
            label_mapping[label] = 0  # Default label if cluster is empty
        else:
            most_common_label = Counter(true_labels_in_cluster).most_common(1)[0][0]
            label_mapping[label] = most_common_label
    return label_mapping

# Map cluster labels to true labels
label_mapping = map_clusters_to_labels(y_test.values, y_pred_clusters)
print(f"Cluster to label mapping: {label_mapping}")

# Apply the mapping to the predicted clusters
y_pred = np.array([label_mapping[label] for label in y_pred_clusters])

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=1, average='binary')
recall = recall_score(y_test, y_pred, pos_label=1, average='binary')
f1 = f1_score(y_test, y_pred, pos_label=1.0, average='binary')

print("Model performance:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Visualization: Confusion Matrix
print("Generating confusion matrix...")
conf_matrix = confusion_matrix(y_test, y_pred, labels=[1, 0])
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
