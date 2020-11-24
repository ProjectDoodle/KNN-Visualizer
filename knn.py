# Standard powerful python tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# Used for splitting the data
from sklearn.model_selection import train_test_split
# Scalar preprocesser used to deal with bias
from sklearn.preprocessing import StandardScaler
# Classifier we're using
from sklearn.neighbors import KNeighborsClassifier
# Testing metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from numpy import random

def euclidean_distance(x, y):
    distance = 0.0
    for i in range(len(x) - 1):
        distance += (x[i] - y[i]) ** 2
    return math.sqrt(distance)

# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors

def main():
    #200 elements, normal dist with mean = 15, sd = 2
    z = 15 + 2*random.randn(200)

    plt.plot(z)
    plt.show(z)

if __name__=="__main__":
    main()