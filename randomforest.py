
import numpy as np
import time
import csv
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.colors as colors
from itertools import cycle
from sklearn.metrics import normalized_mutual_info_score as nmi
import networkx as nx


def scoreForest(estimator, X, y):
    score = estimator.oob_score_
    print "oob_score_:", score
    return score
def rx1():
    reprsn = np.loadtxt('wine/wine_edgelist.txt')
    return reprsn
def ry1():
    labels = np.loadtxt('wine/wine_category.txt')
    return labels
def rx2():
    reprsn = np.loadtxt('data/wiki/wiki_edgelist.txt')
    return reprsn
def ry2():
    labels = np.loadtxt('data/wiki/wiki_category.txt')

    return labels
def rx3():
    reprsn = np.loadtxt('data/cora/cora_edgelist.txt')
    return reprsn
def ry3():
    labels = np.loadtxt('data/cora/cora_category.txt')
    return labels
def rx4():
    #G = nx.read_gml('..\gml\karate.gml')
    reprsn = np.loadtxt('data/karate/karate_edgelist.txt')
    return reprsn
def ry4():
    labels = np.loadtxt('data/karate/karate_category.txt')
    return labels
def cluster(o, n_clusters=3):
    df = pd.read_pickle(o)
    reprsn = df['embedding'].values
    node_idx = df['node_id'].values
    labels = ry1()
    reprsn = [np.asarray(row, dtype='float32') for row in reprsn]
    reprsn = np.array(reprsn, dtype='float32')
    true_labels = [labels[int(node)] for node in node_idx]
    data=reprsn
    km = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    km.fit(data)

    km_means_labels = km.labels_
    km_means_cluster_centers = km.cluster_centers_
    km_means_labels_unique = np.unique(km_means_labels)

    '''colors_ = cycle(colors.cnames.keys())

    m,initial_dim = np.shape(data)
    data_2 = tsne(data, 2, initial_dim, 30)

    plt.figure(figsize=(12, 6))
    plt.scatter(data_2[:, 0], data_2[:, 1], c=true_labels)
    plt.title('True Labels')
    plt.show()'''

    nmiv=nmi(true_labels,km_means_labels)
    print '                 NMI value          ',nmiv

    return nmiv#cal the accuracy of valdata
def cluster0( n_clusters=3):
    labels = ry1()
    true_labels = np.asarray(labels, dtype='int32')
    data=rx1()
    km = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    km.fit(data)

    km_means_labels = km.labels_
    km_means_cluster_centers = km.cluster_centers_
    km_means_labels_unique = np.unique(km_means_labels)

    colors_ = cycle(colors.cnames.keys())


    nmiv=nmi(true_labels,km_means_labels)
    print 'nmi value          ',nmiv

    return nmiv#cal the accuracy of valdata

def rf(o):
    df = pd.read_pickle(o)
    reprsn = df['embedding'].values
    node_idx = df['node_id'].values
    labels = ry1()

    reprsn = [np.asarray(row, dtype='float32') for row in reprsn]
    reprsn = np.array(reprsn, dtype='float32')
    true_labels = [labels[int(node)] for node in node_idx]
    true_labels = np.asarray(true_labels, dtype='int32')
    l=len(reprsn)
    print l
    split=0.9
    train_n=int(split*l)
    train_X = reprsn[0:train_n,:]
    train_y = labels[0:train_n]

    X=reprsn
    y=labels

    print "Rough fitting a RandomForest to determine feature importance..."
    forest = RandomForestClassifier(oob_score=True, n_estimators=10)
    forest.fit(train_X, train_y)

    print "\nFitting model 5 times to get mean OOB score using full training data with class weights..."
    test_scores = []
    # Using the optimal parameters, predict the survival of the labeled test set 10 times
    for i in range(5):
        forest.fit(X,y)
        print "OOB:", forest.oob_score_
        test_scores.append(forest.oob_score_)
    oob = ("%.3f" % (np.mean(test_scores))).lstrip('0')
    oob_std = ("%.3f" % (np.std(test_scores))).lstrip('0')
    oob_lower = ("%.3f" % (np.mean(test_scores) - np.std(test_scores))).lstrip('0')
    print "OOB Mean:", oob, "and stddev:", oob_std
    print "Est. correctly identified test examples rate:", np.mean(test_scores)



    return float(oob)
