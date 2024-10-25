import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# Standardize the data and perform PCA to reduce to 2 dimensions for visualization
def prepare_data_for_clustering(df):
    # Drop non-numeric columns (cuisine, recipeName)
    X = df.drop(['cuisine'], axis=1, errors='ignore')
    # Standardize the data
    X_scaled = StandardScaler().fit_transform(X)
    # Perform PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, df.index  # Return index for labeling

# Function to plot the results of clustering
def plot_clustering_results(X_pca, labels, title, labels_index):
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    palette = sns.color_palette("hsv", len(unique_labels))
    
    # Assign color based on cluster label, excluding noise
    colors = [palette[label] for label in labels]
    
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, s=50, edgecolor='k')

    # Add legend
    handles, _ = scatter.legend_elements()
    labels_list = [f"Cluster {l}" for l in unique_labels]
    plt.legend(handles, labels_list)

    # Proper labeling
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True)
    plt.show()

# DBSCAN clustering without noise
def dbscan_clustering(X_pca, labels_index):
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(X_pca)
    
    # Remove noise by reassigning noise points to nearest cluster
    if -1 in labels:
        from sklearn.neighbors import NearestNeighbors
        neighbors = NearestNeighbors(n_neighbors=1).fit(X_pca[labels != -1])
        noise_indices = (labels == -1)
        _, nearest_clusters = neighbors.kneighbors(X_pca[noise_indices])
        labels[noise_indices] = labels[nearest_clusters.flatten()]

    plot_clustering_results(X_pca, labels, "DBSCAN Clustering (Without Noise)", labels_index)

# Hierarchical clustering
def hierarchical_clustering(X_pca, labels_index):
    Z = linkage(X_pca, method='ward')
    dendrogram(Z, truncate_mode='level', p=5, labels=labels_index)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Recipe Index')
    plt.ylabel('Distance')
    plt.show()

    cluster_model = AgglomerativeClustering(n_clusters=4)
    labels = cluster_model.fit_predict(X_pca)
    plot_clustering_results(X_pca, labels, "Hierarchical Clustering (K=4)", labels_index)

# GMM clustering
def gmm_clustering(X_pca, labels_index):
    gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
    gmm.fit(X_pca)
    labels = gmm.predict(X_pca)
    plot_clustering_results(X_pca, labels, "GMM Clustering (K=4)", labels_index)

if __name__ == '__main__':
    # Load dataset for flavors
    yum_flavor = pd.read_pickle('data/yum_flavor.pkl')

    # Prepare the data for clustering
    X_pca, labels_index = prepare_data_for_clustering(yum_flavor)

    # Apply DBSCAN clustering (without noise)
    dbscan_clustering(X_pca, labels_index)

    # Apply Hierarchical clustering
    hierarchical_clustering(X_pca, labels_index)

    # Apply GMM clustering
    gmm_clustering(X_pca, labels_index)
