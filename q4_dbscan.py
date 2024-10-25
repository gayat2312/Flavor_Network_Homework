import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Function to perform DBSCAN clustering
def dbscan_clustering(df):
    X = df.drop(['cuisine', 'recipeName'], axis=1, errors='ignore')
    X = StandardScaler().fit_transform(X)

    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(X)
    silhouette = silhouette_score(X, labels) if len(set(labels)) > 1 else -1

    print(f"DBSCAN Silhouette Score: {silhouette:.2f}")
    return labels

# Function to perform Hierarchical Clustering
def hierarchical_clustering(df, num_clusters=3):
    X = df.drop(['cuisine', 'recipeName'], axis=1, errors='ignore')
    X = StandardScaler().fit_transform(X)

    # Compute the linkage matrix
    Z = linkage(X, method='ward')

    # Create flat clusters
    clusters = fcluster(Z, num_clusters, criterion='maxclust')

    # Create a dendrogram
    plt.figure(figsize=(10, 6))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    dendrogram(Z, truncate_mode='level', p=3)
    plt.axhline(y=6, color='r', linestyle='--')  # Example line for cutting at a specific distance
    plt.show()

    return clusters

# Function to perform Gaussian Mixture Models (GMM) clustering
def gmm_clustering(df, n_components=3):
    X = df.drop(['cuisine', 'recipeName'], axis=1, errors='ignore')
    X = StandardScaler().fit_transform(X)

    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X)
    labels = gmm.predict(X)
    silhouette = silhouette_score(X, labels) if len(set(labels)) > 1 else -1

    print(f"GMM Silhouette Score: {silhouette:.2f}")
    return labels

if __name__ == '__main__':
    # Load datasets
    yum_ingr = pd.read_pickle('data/yummly_ingr.pkl')
    yum_ingrX = pd.read_pickle('data/yummly_ingrX.pkl')
    yum_flavor = pd.read_pickle('data/yum_flavor.pkl')

    # Prepare dataframes for clustering
    df_ingr = yum_ingrX.copy()
    df_ingr['cuisine'] = yum_ingr['cuisine']
    df_ingr['recipeName'] = yum_ingr['recipeName']

    df_flavor = yum_flavor.copy()
    df_flavor['cuisine'] = yum_ingr['cuisine']
    df_flavor['recipeName'] = yum_ingr['recipeName']

    # DBSCAN Clustering
    print("DBSCAN Clustering for Ingredients:")
    dbscan_labels_ingr = dbscan_clustering(df_ingr)

    print("\nDBSCAN Clustering for Flavor:")
    dbscan_labels_flavor = dbscan_clustering(df_flavor)

    # Hierarchical Clustering
    print("\nHierarchical Clustering for Ingredients:")
    hierarchical_labels_ingr = hierarchical_clustering(df_ingr, num_clusters=3)

    print("\nHierarchical Clustering for Flavor:")
    hierarchical_labels_flavor = hierarchical_clustering(df_flavor, num_clusters=3)

    # GMM Clustering
    print("\nGMM Clustering for Ingredients:")
    gmm_labels_ingr = gmm_clustering(df_ingr, n_components=3)

    print("\nGMM Clustering for Flavor:")
    gmm_labels_flavor = gmm_clustering(df_flavor, n_components=3)
