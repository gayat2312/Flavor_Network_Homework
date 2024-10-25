import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Function to perform PCA and plot 3D scatter
def plot_3d_pca(df, label):
    # Standardize the data
    X = df.drop(['cuisine', 'recipeName'], axis=1, errors='ignore')
    X = StandardScaler().fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=3)  # Keep only the first 3 components for 3D plot
    pca_transformed = pca.fit_transform(X)

    # Create a DataFrame with PCA results
    df_pca = pd.DataFrame(pca_transformed, columns=['Components', 'PC2', 'Explained Overall Cumulative Variance'])
    df_pca['cuisine'] = df['cuisine']

    # Create 3D scatter plot using Plotly
    fig = px.scatter_3d(df_pca, x='Components', y='PC2', z='PC3', color='Explained Overall Cumulative Variance',
                        title=f'3D PCA Plot for {label}', labels={'cuisine': 'Cuisine'})
    
    # Add more interactive features if needed
    fig.update_traces(marker=dict(size=5, opacity=0.8))
    fig.update_layout(scene=dict(xaxis_title='Components', yaxis_title='PC2', zaxis_title='Explained Pverall Cumulative Variance'),
                      legend_title_text='Cuisine', height=700)

    # Show the plot
    fig.show()

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

    # Perform PCA and create 3D scatter plots
    plot_3d_pca(df_ingr, 'Ingredients')  # 3D PCA plot for ingredients
    plot_3d_pca(df_flavor, 'Flavor')     # Already done: 3D PCA plot for flavor
    plot_3d_pca(df_cuisine, 'Cuisines')  # 3D PCA plot for cuisines
