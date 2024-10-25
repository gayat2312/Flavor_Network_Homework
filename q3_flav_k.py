import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Function to perform PCA and K-means clustering
def perform_pca_and_kmeans(df, n_clusters):  
    # Drop non-numeric columns and standardize the data
    X = df.drop(['cuisine', 'recipeName'], axis=1, errors='ignore')
    X_scaled = StandardScaler().fit_transform(X)

    # Perform PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Fit K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, labels)
    print(f"K={n_clusters}, Silhouette Score: {silhouette_avg:.4f}, Inertia: {kmeans.inertia_:.4f}")

    # Plotting K-means clustering results with PCA components
    plt.figure(figsize=(12, 8))
    plt.title(f'K-means Clustering of Flavor (K={n_clusters})')

    # Visualizing the clustering with color mapping
    palette = sns.color_palette("hsv", n_clusters)
    colors = [palette[label] for label in labels]  # Color based on cluster label

    # Scatter plot using PCA components
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, marker='o', edgecolor='k', s=50, alpha=0.6)

    # Plot centroids
    centroids = kmeans.cluster_centers_
    # Transform centroids to PCA space
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='X', s=200, label='Centroids')

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    # Load datasets
    yum_ingr = pd.read_pickle('data/yummly_ingr.pkl')
    yum_ingrX = pd.read_pickle('data/yummly_ingrX.pkl')
    yum_flavor = pd.read_pickle('data/yum_flavor.pkl')

    # Prepare dataframe for clustering
    df_flavor = yum_flavor.copy()
    df_flavor['cuisine'] = yum_ingr['cuisine']  # Assuming cuisine is relevant for the flavor dataset
    df_flavor['recipeName'] = yum_ingr['recipeName']

    # Perform PCA and K-means clustering for K=2, K=3, and K=4
    for k in range(2, 5):  # This will generate K=2, K=3, and K=4
        perform_pca_and_kmeans(df_flavor, n_clusters=k)
