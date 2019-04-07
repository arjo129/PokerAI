import pickle
import numpy as np
from multiprocessing import Pool

cards = pickle.load(open("flop_table.card","rb"))


def get_hist(group):
    hist = {}
    for hole in cards[group]:
        hist[hole] = np.histogram(cards[group][hole], range=(0,1), bins=10)
    return hist


def cosine_distance(hist1, hist2):
    res = 0
    for i in range(len(hist1[0])):
        res += hist1[0][i] * hist2[0][i]
    return res/(sum(map(lambda x: x*x, hist1[0]))*sum(map(lambda x: x*x, hist2[0])))**0.5

def get_distance_matrix(hist):
    arr = np.zeros(shape=(len(hist), len(hist)))
    i,j = 0,0
    card_index = {}
    for k1 in hist:
        card_index[i] = k1 
        print(i)
        j = 0
        for k2 in hist:
            arr[i][j] = acos(cosine_distance(hist[k1], hist[k2]))
            j += 1
        i += 1
    return arr

def workload(i):
    histograms = get_hist(i)
    matrix = get_distance_matrix(histograms)
    pickle.dump(open("group"+str(i)+".distancematrix", "wb+"))

with Pool(NUM_PROCESSES) as pool:
    result = pool.map(workload, list(range(-1,18))


