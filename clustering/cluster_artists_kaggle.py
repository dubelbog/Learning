"""
Cluster artists based on the words in their lyrics
Data from kaggle
Format data with create_data_kaggle.py
"""

__author__ = 'don.tuggener@zhaw.ch'

import json
import re
import seaborn as sns
import numpy
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


def plot_dendrogram(clustered, artists):
    """ Plot a dendrogram from the hierarchical clustering of the artist lyrics """
    # plt.figure(figsize=(25, 10))   # for orientation = 'bottom'|'top'
    plt.figure(figsize=(10, 25))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Distance')  # this' but the label of the whole axis!
    plt.ylabel('Artists')
    plt.tight_layout()
    dendrogram(clustered,
               # leaf_rotation=90.,  # rotates the x axis labels
               leaf_font_size=8.,  # font size for the x axis labels
               labels=artists,
               orientation='left',
               )
    # plt.show() # Instead pf saving
    plt.savefig('dendrogram.svg', bbox_inches='tight')


def words_per_artist(artist_lyrics, lyrics_tfidf_matrix, ix2word, n=10):
    """ 
    For each artist, print the most highly weighted words acc. to TF IDF 
    Print n words that are above the mean weight 
    """
    file = open("words_per_artist.txt", "w+")
    num_iteration = len(artist_lyrics)
    artist_names = list(artist_lyrics.keys())

    for i in range(num_iteration):
        file.write(artist_names[i] + ": \n")
        sorted_matrix_descending = numpy.sort(lyrics_tfidf_matrix[i])[::-1]
        words = set()
        index = 0
        while len(words) < n:
            for pos in numpy.where(lyrics_tfidf_matrix[i] == sorted_matrix_descending[index]):
                for k in pos:
                    words.add(ix2word[k])
                index += 1

        sorted_words = sorted(words)
        [file.write(word + ", ") for word in sorted_words]
        file.write("\n")


def words_per_genres(true_n, centers):
    file_means = open("words_genres.txt", "w+")
    for i in range(true_n):
        file_means.write("genre cluster: " + str(i) + "\n")
        sorted_centers = sorted(centers[i])[::-1]
        words = set()
        pos = 0
        while len(words) < 15:
            index = list(centers[i]).index(sorted_centers[pos])
            words.add(ix2word[index])
            pos += 1

        sorted_words = sorted(words)
        [file_means.write(word + ", ") for word in sorted_words]
        file_means.write("\n")


def elbow_method(x):
    sse = {}
    for ncenters in range(1, 90):
        kmeans = KMeans(n_clusters=ncenters, init='k-means++', random_state=0).fit(x)
        sse[ncenters] = kmeans.inertia_
    plt.title('The Elbow Method')
    plt.xlabel('k')
    plt.ylabel('SSE')
    sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
    plt.savefig('elbow_method.png')
    plt.show()


def plot_pca(lyrics_tfidf_matrix, artist_lyrics):
    x_artists = lyrics_tfidf_matrix
    y_artists = list(artist_lyrics.keys())
    x_std = StandardScaler().fit_transform(x_artists)

    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(x_std)

    with plt.style.context('bmh'):
        plt.figure(figsize=(12, 8), facecolor='#b9c9ba')
        for label, coord in zip(y_artists, coordinates):
            plt.scatter(coord[0], coord[1], alpha=.8)
        #    plt.annotate(label, (coord[0], coord[1]))
        plt.xlabel('PC0')
        plt.ylabel('PC1')
        plt.tight_layout()
        plt.savefig('pca.png')
        plt.show()


def k_means(X, true_n):
    model = KMeans(n_clusters=true_n, init='k-means++', max_iter=150).fit(X)
    y_kmeans = model.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=20, cmap='viridis')
    centers = model.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='grey', s=100, alpha=0.5)
    plt.savefig('k_means.png')
    plt.show()
    return centers


def remove_outliers(artist_lyrics, artist2genre):
    del artist_lyrics['eminem']
    del artist2genre['eminem']
    del artist_lyrics['celine-dion']
    del artist2genre['celine-dion']
    del artist_lyrics['edith-piaf']
    del artist2genre['edith-piaf']
    # second group of outliers
    del artist_lyrics['e-40']
    del artist2genre['e-40']
    del artist_lyrics['cseh-tamxi-xi']
    del artist2genre['cseh-tamxi-xi']
    del artist_lyrics['frank-zappa']
    del artist2genre['frank-zappa']
    del artist_lyrics['chamillionaire']
    del artist2genre['chamillionaire']
    del artist_lyrics['busta-rhymes']
    del artist2genre['busta-rhymes']
    del artist_lyrics['game']
    del artist2genre['game']
    del artist_lyrics['cradle-of-filth']
    del artist2genre['cradle-of-filth']


if __name__ == '__main__':
    print('Loading data')
    artist2genre = json.load(open('data/artist2genre_kaggle.json', 'r', encoding='utf-8'))
    artist_lyrics = json.load(open('data/artist_lyrics_kaggle.json', 'r', encoding='utf-8'))
    remove_outliers(artist_lyrics, artist2genre)
    # Custom tokenization to remove numbers etc.
    lyrics = [' '.join(re.findall('[A-Za-z]+', l)) for l in artist_lyrics.values()]

    print('Vectorizing with TF IDF')
    # Vectorize the song lyrics
    # TODO implement; create lyrics_tfidf_matrix (artist/word matrix) and ix2word (dict that maps word IDs to words)
    # works well if enter method remove_outliers is used. The model is very robust and stable
    # the param max_df has not a big influence on results anymore
    vectorizer = TfidfVectorizer(stop_words='english', sublinear_tf=True)
    # works well if the second group of outliers is commented out / but the model is overfitted
    # every small changes has a big influence on the dendrogram classification
    # vectorizer = TfidfVectorizer(max_df=0.1132, stop_words='english', sublinear_tf=True)
    lyrics_tfidf_matrix = vectorizer.fit_transform(lyrics).toarray()
    ix2word = dict(enumerate(vectorizer.get_feature_names()))

    print('Distinct words per artist')
    words_per_artist(artist_lyrics, lyrics_tfidf_matrix, ix2word)

    print('Clustering')
    # TODO call SciPy's hierarchical clustering
    genres_dict = {}
    for artist in artist_lyrics.keys():
        genres_dict[artist2genre[artist]] = artist

    plot_pca(lyrics_tfidf_matrix, artist_lyrics)

    X = lyrics_tfidf_matrix
 #   elbow_method(X)

    true_n = len(genres_dict)
    centers = k_means(X, true_n)
    words_per_genres(true_n, centers)

    print('Plotting')
    mapping_artists_genre = [artist2genre[a] + ' - ' + a for a in list(artist_lyrics.keys())]
    clustered = linkage(lyrics_tfidf_matrix, method='ward')
    plot_dendrogram(clustered, mapping_artists_genre)
