import numpy as np
import pickle
from sklearn.manifold import TSNE

def euclidean_dist(vec1, vec2):
    return np.sqrt(np.sum((vec1-vec2)**2))

def find_closest(word_index, vectors):
    query_vector = vectors[word_index]
    distance = []
    for index, vector in enumerate(vectors):
        dist = euclidean_dist(vector, query_vector)
        distance.append((dist, index))
    distance = sorted(distance, key = lambda x: x[0])
    return distance

def get_key(val, my_dict): 
    for key, value in my_dict.items(): 
         if val == value: 
             return key 


def doit(picke_w, filename):
    embed, wordIndex, wordDict = pickle.load(picke_w)
    # distance = find_closest(1, embed)
    # print(distance)
    # print(get_key(1, wordIndex))
    # for i in distance[1:11]:
    #     print(get_key(i[1], wordIndex))
    keys = range(20)
    embedding_clusters = []
    word_clusters = []
    for word in keys:
        embeddings = []
        words = []
        for similar_word in find_closest(word, embed)[1:11]:
            words.append(similar_word[1])
            embeddings.append(embed[similar_word[1]])
        # print(words)
        # print(embeddings)
        embedding_clusters.append(embeddings)
        word_clusters.append(words)
    # print(word_clusters)
    # print(embedding_clusters)
    print("EMBED DONE")
    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
    embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    # % matplotlib inline


    def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename=None):
        plt.figure(figsize=(16, 9))
        colors = cm.rainbow(np.linspace(0, 1, len(labels)))
        for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
            x = embeddings[:, 0]
            y = embeddings[:, 1]
            plt.scatter(x, y, c=color, alpha=a, label=label)
            for i, word in enumerate(words):
                plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                             textcoords='offset points', ha='right', va='bottom', size=8)
        plt.legend(loc=4)
        plt.title(title)
        plt.grid(True)
        # if filename:
        #     plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
        # plt.show()
        plt.savefig(filename+".png")
        print("SAVED IMAGE",filename)


    tsne_plot_similar_words('Similar words from ABC', keys, embeddings_en_2d, word_clusters, 0.7, filename)


import os 
curList = os.listdir("modelsnew")
for i in curList:
    print(i)
    picke_w = open("modelsnew/"+i,"rb")
    doit(picke_w, i)
