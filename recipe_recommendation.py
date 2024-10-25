import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# This function prints out dishes similar to the query, either including or excluding dishes from its own cuisine, based on 'similar_cuisine'
def find_dishes(idx, similar_cuisine=False):
    cuisine = yum_ingr2.iloc[idx]['cuisine']
    print('Dishes similar to', yum_ingr2.iloc[idx]['recipeName'], '(' + cuisine + ')')
    match = yum_ingr2.iloc[yum_cos[idx].argsort()[-21:-1]][::-1]

    if not similar_cuisine:
        submatch = match[match['cuisine'] != cuisine]
    else:
        submatch = match
        
    print()
    for i in submatch.index:
        print(submatch.iloc[i]['recipeName'], '(' + submatch.iloc[i]['cuisine'] + ')', '(ID:' + str(i) + ')')


# This function plots the top-20 dishes similar to the query
def plot_similar_dishes(idx, xlim):
    match = yum_ingr2.iloc[yum_cos[idx].argsort()[-21:-1]][::-1]
    newidx = match.index.values
    match['cosine'] = yum_cos[idx][newidx]
    match['rank'] = range(1, 1 + len(newidx))

    label1, label2 = [], []
    for i in match.index:
        label1.append(match.iloc[i]['cuisine'])
        label2.append(match.iloc[i]['recipeName'])

    fig = plt.figure(figsize=(10, 10))
    ax = sns.stripplot(y='rank', x='cosine', data=match, jitter=0.05,
                       hue='cuisine', size=15, orient="h")
    ax.set_title(yum_ingr2.iloc[idx]['recipeName'] + ' (' + yum_ingr2.iloc[idx]['cuisine'] + ')', fontsize=18)
    ax.set_xlabel('Flavor cosine similarity', fontsize=18)
    ax.set_ylabel('Rank', fontsize=18)
    ax.yaxis.grid(color='white')
    ax.xaxis.grid(color='white')

    for label, y, x in zip(label2, match['rank'], match['cosine']):
        ax.text(x + 0.001, y - 1, label, ha='left')
    ax.legend(loc='lower right', prop={'size': 14})
    ax.set_ylim([20, -1])
    ax.set_xlim(xlim)


if __name__ == '__main__':
    yum_ingr = pd.read_pickle('data/yummly_ingr.pkl')
    yum_ingrX = pd.read_pickle('data/yummly_ingrX.pkl')
    yum_tfidf = pd.read_pickle('data/yumm_tfidf.pkl')
    
    # Calculate cosine similarity
    yum_cos = cosine_similarity(yum_tfidf)
    
    # Reset index for yum_ingr
    yum_ingr2 = yum_ingr.reset_index(drop=True)

    # Plot similar dishes for Fettuccine Bolognese
    idx = 3900
    xlim = [0.91, 1.0]
    plot_similar_dishes(idx, xlim)

    # Plot similar dishes for chicken tikka masala
    idx = 3315
    xlim = [0.88, 1.02]
    plot_similar_dishes(idx, xlim)
