import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage

# Standardize the data and perform PCA to reduce to 2 dimensions for visualization
def prepare_data_for_clustering(df):
    # Drop non-numeric columns (cuisine, recipeName)
    X = df.drop(['cuisine', 'recipeName'], axis=1, errors='ignore')
    # Standardize the data
    X_scaled = StandardScaler().fit_transform(X)
    # Perform PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, df['recipeName']  # Return recipe names for labeling

# Function to plot the results of clustering
def plot_clustering_results(X_pca, labels, title, recipe_names):
    plt.figure(figsize=(10, 8))
    palette = sns.color_palette("hsv", np.unique(labels).max() + 1)
    colors = [palette[label] if label != -1 else (0, 0, 0) for label in labels]  # Black for noise points in DBSCAN
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, s=50, edgecolor='k')

    # Add legend
    handles, _ = scatter.legend_elements()
    unique_labels = np.unique(labels)
    labels_list = [f"Cluster {l}" if l != -1 else "Noise" for l in unique_labels]
    plt.legend(handles, labels_list)

    # Proper labeling
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True)
    plt.show()

# DBSCAN clustering with noise reassignment
def dbscan_clustering(X_pca, recipe_names):
    dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust eps and min_samples as needed
    labels = dbscan.fit_predict(X_pca)

    # If there is noise (-1 label), reassign noise points to nearest cluster
    if -1 in labels:
        neighbors = NearestNeighbors(n_neighbors=1).fit(X_pca[labels != -1])
        noise_indices = (labels == -1)
        _, nearest_clusters = neighbors.kneighbors(X_pca[noise_indices])
        labels[noise_indices] = labels[nearest_clusters.flatten()]

    plot_clustering_results(X_pca, labels, "DBSCAN Clustering (Without Noise)", recipe_names)

# Hierarchical clustering
def hierarchical_clustering(X_pca, recipe_names):
    # Plot the dendrogram with recipe names as labels
    plt.figure(figsize=(10, 8))
    Z = linkage(X_pca, method='ward')  # Ward's method for hierarchical clustering
    dendrogram(Z, truncate_mode='level', p=5, labels=recipe_names.values)
    plt.title('Hierarchical Clustering Dendrogram (Ingredients)')
    plt.xlabel('Recipe Names')
    plt.ylabel('Distance')
    plt.show()

    # Plot clusters based on hierarchical clustering (truncated)
    cluster_model = AgglomerativeClustering(n_clusters=4)
    labels = cluster_model.fit_predict(X_pca)
    plot_clustering_results(X_pca, labels, "Hierarchical Clustering (K=4)", recipe_names)

# GMM clustering
def gmm_clustering(X_pca, recipe_names):
    gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
    gmm.fit(X_pca)
    labels = gmm.predict(X_pca)
    plot_clustering_results(X_pca, labels, "GMM Clustering (K=4)", recipe_names)

if __name__ == '__main__':
    # Load dataset
    yum_ingr = pd.read_pickle('data/yummly_ingr.pkl')
    yum_ingrX = pd.read_pickle('data/yummly_ingrX.pkl')

    # Prepare dataframe for clustering (ingredients only)
    df_ingr = yum_ingrX.copy()
    df_ingr['cuisine'] = yum_ingr['cuisine']
    df_ingr['recipeName'] = yum_ingr['recipeName']

    # Prepare the data for clustering
    X_pca, recipe_names = prepare_data_for_clustering(df_ingr)

    # Apply DBSCAN clustering (without noise)
    dbscan_clustering(X_pca, recipe_names)

    # Apply Hierarchical clustering
    hierarchical_clustering(X_pca, recipe_names)

    # Apply GMM clustering
    gmm_clustering(X_pca, recipe_names)
