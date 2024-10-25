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

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
        
        # Calculate silhouette score only if k > 1
        if k > 1:
            score = silhouette_score(X, kmeans.labels_)
            silhouette_scores.append(score)

    # Plot Elbow Method and Silhouette Scores
    plt.figure(figsize=(12, 5))

    # Elbow Method
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_k + 1), inertia, marker='o', label=label, color=color)
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters K')
    plt.ylabel('Inertia')
    plt.xticks(range(2, max_k + 1))
    plt.grid()
    plt.legend()

    # Silhouette Scores
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_k + 1), silhouette_scores, marker='o', label=label, color=color)
    plt.title('Silhouette Scores')
    plt.xlabel('Number of Clusters K')
    plt.ylabel('Silhouette Score')
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

    # Prepare dataframe for Cuisines (similar to ingredients and flavors)
    df_cuisine = df_ingr.groupby('cuisine').mean().reset_index()  # Example aggregation for cuisines

    # Perform K-means analysis for Cuisines
    print("K-means Analysis for Cuisines:")
    kmeans_analysis(df_cuisine, label='Cuisines', color=colors['Cuisines'], max_k=10)
