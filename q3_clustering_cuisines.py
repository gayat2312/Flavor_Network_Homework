import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool

# Function for t-SNE clustering and plotting
def tsne_cluster_cuisine(df):
    sublist = df['cuisine'].unique()  # Get all unique cuisines
    lenlist = [0]
    df_sub = df[df['cuisine'] == sublist[0]]
    lenlist.append(df_sub.shape[0])
    
    for cuisine in sublist[1:]:
        temp = df[df['cuisine'] == cuisine]
        df_sub = pd.concat([df_sub, temp], axis=0, ignore_index=True)
        lenlist.append(df_sub.shape[0])
    
    df_X = df_sub.drop(['cuisine', 'recipeName'], axis=1)
    print(df_X.shape, lenlist)

    dist = squareform(pdist(df_X, metric='cosine'))
    tsne = TSNE(metric='precomputed').fit_transform(dist)

    palette = sns.color_palette("hls", len(sublist))
    plt.figure(figsize=(10, 10))
    
    for i, cuisine in enumerate(sublist):
        plt.scatter(tsne[lenlist[i]:lenlist[i + 1], 0],
                    tsne[lenlist[i]:lenlist[i + 1], 1],
                    c=palette[i], label=cuisine)
    
    plt.legend()
    plt.title("t-SNE Clustering of Cuisines")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.show()

# Interactive plot with Bokeh
def plot_bokeh(df, filename):
    sublist = df['cuisine'].unique()  # Get all unique cuisines
    lenlist = [0]
    df_sub = df[df['cuisine'] == sublist[0]]
    lenlist.append(df_sub.shape[0])
    
    for cuisine in sublist[1:]:
        temp = df[df['cuisine'] == cuisine]
        df_sub = pd.concat([df_sub, temp], axis=0, ignore_index=True)
        lenlist.append(df_sub.shape[0])
    
    df_X = df_sub.drop(['cuisine', 'recipeName'], axis=1)
    print(df_X.shape, lenlist)

    dist = squareform(pdist(df_X, metric='cosine'))
    tsne = TSNE(metric='precomputed').fit_transform(dist)

    # Create color mapping
    palette = sns.color_palette("hls", len(sublist))
    colors = []
    for i in range(len(sublist)):
        for j in range(lenlist[i + 1] - lenlist[i]):
            colors.append(palette[i])

    # Plot with Bokeh
    output_file(filename)
    source = ColumnDataSource(
        data=dict(x=tsne[:, 0], y=tsne[:, 1],
                  cuisine=df_sub['cuisine'],
                  recipe=df_sub['recipeName']))

    hover = HoverTool(tooltips=[
        ("cuisine", "@cuisine"),
        ("recipe", "@recipe")])

    p = figure(plot_width=1000, plot_height=1000, tools=[hover],
               title="Flavor Clustering")

    p.circle('x', 'y', size=10, source=source, fill_color=colors)

    show(p)

if __name__ == '__main__':
    yum_ingr = pd.read_pickle('data/yummly_ingr.pkl')
    yum_ingrX = pd.read_pickle('data/yummly_ingrX.pkl')
    yum_tfidf = pd.read_pickle('data/yum_tfidf.pkl')

    # Prepare data for t-SNE clustering with ingredients
    df_ingr = yum_ingrX.copy()
    df_ingr['cuisine'] = yum_ingr['cuisine']
    df_ingr['recipeName'] = yum_ingr['recipeName']
    tsne_cluster_cuisine(df_ingr)

    # Prepare data for t-SNE clustering with flavor
    df_flavor = yum_tfidf.copy()
    df_flavor['cuisine'] = yum_ingr['cuisine']
    df_flavor['recipeName'] = yum_ingr['recipeName']
    tsne_cluster_cuisine(df_flavor)

    # Create interactive plotting with Bokeh
    plot_bokeh(df_flavor, 'flavor_clustering.html')
    plot_bokeh(df_ingr, 'ingredient_clustering.html')
