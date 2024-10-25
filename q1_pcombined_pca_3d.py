import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Function to perform PCA and return the transformed data with labels
def perform_pca_and_get_transformed(df, label):
    # Standardize the data
    X = df.drop(['cuisine', 'recipeName'], axis=1, errors='ignore')
    X = StandardScaler().fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=3)
    pca_transformed = pca.fit_transform(X)

    # Create a DataFrame with PCA results and cumulative variance
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    df_pca = pd.DataFrame(pca_transformed, columns=['Componenets', 'PC2', 'Explained Overall Cumulative Variance'])
    df_pca['cuisine'] = df['cuisine']
    df_pca['label'] = label  # Add label for distinguishing between datasets
    df_pca['cumulative_variance'] = explained_variance[2]  # Cumulative variance after 3 components

    return df_pca, explained_variance

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

    # Perform PCA for each dataset and get the transformed data
    pca_ingr, ingr_variance = perform_pca_and_get_transformed(df_ingr, 'Ingredients')
    pca_flavor, flavor_variance = perform_pca_and_get_transformed(df_flavor, 'Flavor')
    pca_cuisine, cuisine_variance = perform_pca_and_get_transformed(df_cuisine, 'Cuisines')

    # Combine all PCA results into one DataFrame
    df_combined = pd.concat([pca_ingr, pca_flavor, pca_cuisine], ignore_index=True)

    # Create 3D scatter plot for the combined PCA results
    fig = px.scatter_3d(df_combined, x='Componenets', y='PC2', z='Explained Overall Cumulative Variance', color='label',
                        hover_data=['cuisine'], 
                        title='Combined 3D PCA Plot for Ingredients, Flavor, and Cuisines',
                        labels={'label': 'Dataset', 'cuisine': 'Cuisine'})

    # Customize the layout for better visualization
    fig.update_traces(marker=dict(size=5, opacity=0.7))
    fig.update_layout(scene=dict(xaxis_title='Componenets', yaxis_title='PC2', zaxis_title='Explained Overall Cumulative Variance'),
                      legend_title_text='Dataset', height=800)

    # Show the plot
    fig.show()

    # Print cumulative variance for each dataset
    print("Cumulative Variance Explained after 3 components:")
    print(f"Ingredients: {ingr_variance[2]:.4f}")
    print(f"Flavor: {flavor_variance[2]:.4f}")
    print(f"Cuisines: {cuisine_variance[2]:.4f}")
