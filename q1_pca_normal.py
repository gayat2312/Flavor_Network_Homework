###
#PCA ANALYIS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Function to perform PCA and plot cumulative explained variance
def perform_pca_and_plot(df, label, ax):
    # Standardize the data
    X = df.drop(['cuisine', 'recipeName'], axis=1, errors='ignore')  # Ignore errors if not present
    X = StandardScaler().fit_transform(X)

    # Apply PCA
    pca = PCA()
    pca.fit(X)

    # Explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # Plot cumulative explained variance
    ax.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, label=label)
    
    # Print explained variance for each component
    print(f"{label} Explained Variance Ratios:")
    for i, variance in enumerate(explained_variance):
        print(f"Component {i + 1}: {variance * 100:.2f}%")
    
    return cumulative_variance

if __name__ == '__main__':
    # Load datasets
    yum_ingr = pd.read_pickle('data/yummly_ingr.pkl')
    yum_ingrX = pd.read_pickle('data/yummly_ingrX.pkl')
    yum_flavor = pd.read_pickle('data/yum_flavor.pkl')

    # Prepare dataframes for PCA
    df_ingr = yum_ingrX.copy()
    df_ingr['cuisine'] = yum_ingr['cuisine']
    df_ingr['recipeName'] = yum_ingr['recipeName']

    df_flavor = yum_flavor.copy()
    df_flavor['cuisine'] = yum_ingr['cuisine']
    df_flavor['recipeName'] = yum_ingr['recipeName']

    # Create a cuisine-level DataFrame for PCA
    df_cuisine = df_ingr.groupby('cuisine').mean().reset_index()  # Group by cuisine and average

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Perform PCA for Ingredients, Flavor, and Cuisines
    perform_pca_and_plot(df_ingr, 'Ingredients', ax)
    perform_pca_and_plot(df_flavor, 'Flavor', ax)
    perform_pca_and_plot(df_cuisine, 'Cuisines', ax)

    # Add details to the plot
    ax.set_title('Cumulative Explained Variance by PCA Components')
    ax.set_xlabel('Number of Principal Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.axhline(y=0.95, color='r', linestyle='--', label='95% Variance Threshold')
    ax.legend()
    ax.grid()

    plt.tight_layout()  # Adjust layout to prevent clipping of titles/labels
    plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Function to perform PCA and plot cumulative explained variance
def perform_pca_and_plot(df, label, ax):
    # Standardize the data
    X = df.drop(['cuisine', 'recipeName'], axis=1)  # Drop non-numeric columns
    X = StandardScaler().fit_transform(X)

    # Apply PCA
    pca = PCA()
    pca.fit(X)

    # Explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # Plot cumulative explained variance
    ax.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, label=label)
    
    # Print explained variance for each component
    print(f"{label} Explained Variance Ratios:")
    for i, variance in enumerate(explained_variance):
        print(f"Component {i + 1}: {variance * 100:.2f}%")
    
    return cumulative_variance

if __name__ == '__main__':
    # Load datasets
    yum_ingr = pd.read_pickle('data/yummly_ingr.pkl')
    yum_ingrX = pd.read_pickle('data/yummly_ingrX.pkl')
    yum_flavor = pd.read_pickle('data/yum_flavor.pkl')

    # Prepare dataframes
    df_ingr = yum_ingrX.copy()
    df_ingr['cuisine'] = yum_ingr['cuisine']
    df_ingr['recipeName'] = yum_ingr['recipeName']
    df_flavor = yum_flavor.copy()
    df_flavor['cuisine'] = yum_ingr['cuisine']
    df_flavor['recipeName'] = yum_ingr['recipeName']

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Perform PCA for Ingredients, Flavor, and Cuisines
    cumul_var_ingr = perform_pca_and_plot(df_ingr, 'Ingredients', ax)
    cumul_var_flavor = perform_pca_and_plot(df_flavor, 'Flavor', ax)

    # To add PCA for cuisines, you may need a cuisine-specific DataFrame. 
    # Assuming you have a function or dataset that allows you to extract that.
    # Here, we'll create a dummy DataFrame for demonstration.
    # Replace this with the actual cuisine DataFrame if available.
    df_cuisine = df_ingr.groupby('cuisine').mean()  # Example aggregation
    cumul_var_cuisine = perform_pca_and_plot(df_cuisine, 'Cuisines', ax)

    # Add details to the plot
    ax.set_title('Cumulative Explained Variance by PCA Components')
    ax.set_xlabel('Number of Principal Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.axhline(y=0.95, color='r', linestyle='--', label='95% Variance Threshold')
    ax.legend()
    ax.grid()
    plt.show()