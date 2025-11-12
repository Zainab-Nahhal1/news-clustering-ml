import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)

def main():
    # Load sample data (replace with your dataset)
    data = pd.read_csv('sample/news_data_sample.tsv', sep='\t')
    data.dropna(subset=['text'], inplace=True)
    data['clean_text'] = data['text'].apply(preprocess_text)

    # TF-IDF vectorization
    tfidf = TfidfVectorizer(max_features=1000)
    X = tfidf.fit_transform(data['clean_text']).toarray()

    # Dimensionality reduction with PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    # K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels_kmeans = kmeans.fit_predict(X_pca)

    # Agglomerative clustering
    agg = AgglomerativeClustering(n_clusters=3)
    labels_agg = agg.fit_predict(X_pca)

    # Evaluate clustering with silhouette score
    score_kmeans = silhouette_score(X_pca, labels_kmeans)
    score_agg = silhouette_score(X_pca, labels_agg)

    print(f"Silhouette Score (K-Means): {score_kmeans:.3f}")
    print(f"Silhouette Score (Agglomerative): {score_agg:.3f}")

    # Visualize clustering
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.scatter(X_pca[:,0], X_pca[:,1], c=labels_kmeans, cmap='viridis')
    plt.title('K-Means Clustering')

    plt.subplot(1,2,2)
    plt.scatter(X_pca[:,0], X_pca[:,1], c=labels_agg, cmap='plasma')
    plt.title('Agglomerative Clustering')
    plt.show()

if __name__ == "__main__":
    main()
