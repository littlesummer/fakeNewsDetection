# Install necessary libraries
# !pip install pandas nltk scikit-learn matplotlib seaborn gensim graphviz

import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
import numpy as np
import graphviz
from sklearn.tree import export_graphviz

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load data and drop rows with NaN values
# data = pd.read_csv('/content/fake_news.csv', sep=',', on_bad_lines='skip', quoting=3, low_memory=False)
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

from sklearn.svm import SVC

# Train the SVM model
print("Training the SVM model with Word2Vec features...")
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_vect, y_train)
print("Model training completed.")

# Predict and calculate performance metrics
print("Evaluating model performance with SVM...")
y_pred = svm_model.predict(X_test_vect)

# Assuming 1 represents "fake" and 0 represents "real"
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=1, average='binary')
recall = recall_score(y_test, y_pred, pos_label=1, average='binary')
f1 = f1_score(y_test, y_pred, pos_label=1.0, average='binary')

print("SVM Model performance:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Visualization: Confusion Matrix
print("Generating confusion matrix for SVM...")
conf_matrix = confusion_matrix(y_test, y_pred, labels=[1, 0])
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Fake', 'Real'], yticklabels=['Real', 'Real'])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix (SVM)")
plt.show()
