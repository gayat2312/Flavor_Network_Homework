import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Function to calculate Silhouette and Elbow scores for each dataset
def evaluate_kmeans(df, label, max_k=10):
    # Drop non-numeric columns and standardize the data
    X = df.drop(['cuisine', 'recipeName'], axis=1, errors='ignore')
    X_scaled = StandardScaler().fit_transform(X)

    # Store inertia (for elbow method) and silhouette scores
    inertia = []
    silhouette_scores = []

    # Iterate over K values from 2 to max_k
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        
        # Inertia for Elbow method
        inertia.append(kmeans.inertia_)
        
        # Silhouette Score
        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(score)
        print(f"{label} - K={k}, Silhouette Score: {score:.4f}, Inertia: {kmeans.inertia_:.4f}")

    # Plot Elbow Method and Silhouette Scores
    plt.figure(figsize=(12, 5))

    # Elbow Method (Inertia)
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_k + 1), inertia, marker='o', color='blue')
    plt.title(f'{label} - Elbow Method')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.grid()

    # Silhouette Scores
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_k + 1), silhouette_scores, marker='o', color='orange')
    plt.title(f'{label} - Silhouette Scores')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.grid()

    plt.tight_layout()
    plt.show()

    # Get the best K (highest silhouette score)
    best_k = np.argmax(silhouette_scores) + 2  # +2 because K starts at 2
    print(f"Best K for {label} based on Silhouette Score: {best_k}")
    return best_k

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

    # Cuisine-level data (aggregated by cuisine)
    df_cuisine = df_ingr.groupby('cuisine').mean().reset_index()

    # Perform evaluation for Ingredients, Flavor, and Cuisines
    print("\nEvaluating Ingredients Clustering:")
    best_k_ingr = evaluate_kmeans(df_ingr, label='Ingredients', max_k=10)

    print("\nEvaluating Flavor Clustering:")
    best_k_flavor = evaluate_kmeans(df_flavor, label='Flavor', max_k=10)

    print("\nEvaluating Cuisine Clustering:")
    best_k_cuisine = evaluate_kmeans(df_cuisine, label='Cuisines', max_k=10)

    # Print summary of best Ks
    print(f"\nBest K for Ingredients: {best_k_ingr}")
    print(f"Best K for Flavor: {best_k_flavor}")
    print(f"Best K for Cuisines: {best_k_cuisine}")
