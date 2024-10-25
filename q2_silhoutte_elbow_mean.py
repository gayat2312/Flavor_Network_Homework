import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Function to perform K-means clustering and analyze K
def kmeans_analysis(df, label, color, max_k=10):
    # Standardize the data
    X = df.drop(['cuisine', 'recipeName'], axis=1, errors='ignore')  # Drop non-numeric columns
    X = StandardScaler().fit_transform(X)

    # Elbow method
    inertia = []
    silhouette_scores = []
    silhouette_scores_mean = []

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
        
        # Calculate silhouette score only if k > 1
        if k > 1:
            score = silhouette_score(X, kmeans.labels_)
            silhouette_scores.append(score)
            silhouette_scores_mean.append(np.mean(silhouette_scores))

    # Plot Elbow Method and Silhouette Scores
    plt.figure(figsize=(12, 10))

    # Elbow Method
    plt.subplot(2, 2, 1)
    plt.plot(range(2, max_k + 1), inertia, marker='o', label=label, color=color)
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters K')
    plt.ylabel('Inertia')
    plt.xticks(range(2, max_k + 1))
    plt.grid()
    plt.legend()

    # Silhouette Scores
    plt.subplot(2, 2, 2)
    plt.plot(range(2, max_k + 1), silhouette_scores, marker='o', label=label, color=color)
    plt.title('Silhouette Scores')
    plt.xlabel('Number of Clusters K')
    plt.ylabel('Silhouette Score')
    plt.xticks(range(2, max_k + 1))
    plt.axhline(y=np.mean(silhouette_scores), color='r', linestyle='--', label='Mean Silhouette Score')
    plt.grid()
    plt.legend()

    # Cumulative Mean Silhouette Scores
    plt.subplot(2, 2, 3)
    plt.plot(range(2, max_k + 1), silhouette_scores_mean, marker='o', label=label, color=color)
    plt.title('Cumulative Mean Silhouette Scores')
    plt.xlabel('Number of Clusters K')
    plt.ylabel('Cumulative Mean Silhouette Score')
    plt.xticks(range(2, max_k + 1))
    plt.grid()
    plt.legend()

    # Inertia vs Silhouette Scores with Cluster Dots
    plt.subplot(2, 2, 4)
    plt.scatter(range(2, max_k + 1), silhouette_scores, label='Silhouette Score', color='purple', marker='o')
    plt.plot(range(2, max_k + 1), inertia, label='Inertia', color=color, marker='o', alpha=0.5)
    plt.title('Inertia vs Silhouette Scores')
    plt.xlabel('Number of Clusters K')
    plt.ylabel('Score')
    plt.xticks(range(2, max_k + 1))
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

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

    # Prepare dataframe for Cuisines (similar to ingredients and flavors)
    df_cuisine = df_ingr.groupby('cuisine').mean().reset_index()  # Example aggregation for cuisines

    # Define colors for each analysis
    colors = {
        'Ingredients': 'blue',
        'Flavor': 'orange',
        'Cuisines': 'green'
    }

    # Perform K-means analysis for Ingredients
    print("K-means Analysis for Ingredients:")
    kmeans_analysis(df_ingr, label='Ingredients', color=colors['Ingredients'], max_k=10)
    
    # Perform K-means analysis for Flavor
    print("K-means Analysis for Flavor:")
    kmeans_analysis(df_flavor, label='Flavor', color=colors['Flavor'], max_k=10)

    # Perform K-means analysis for Cuisines
    print("K-means Analysis for Cuisines:")
    kmeans_analysis(df_cuisine, label='Cuisines', color=colors['Cuisines'], max_k=10)
