'''
Authors: Matthew Robinson, Antonio Munoz
Date: 11/23/20
Class: CSE 489 ML
Description: KNN Visualizer
To-Do: Roll our own random points generator
'''



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

from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from numpy import random


'''
Description: Computes the euclidean distance
Param x: x coordinate point
Param y: y coordinate
'''
def euclidean_distance(x, y):
    distance = 0.0
    for i in range(len(x) - 1):
        distance += (x[i] - y[i]) ** 2
    return math.sqrt(distance)


'''
Description: Finds the nearest neighbors
Param train:
Param test_row:
Param num_neighbors:
'''
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


'''
Description: Generate random points
Param num_points: The number of random points to generate

Mostly taken from this (https://stackoverflow.com/questions/19668463/generating-multiple-random-x-y-coordinates-excluding-duplicates)
Will roll our own later, using this for testing purposes
'''
def generate_points(num_points):
    radius = 200
    rangeX = (0, 100)
    rangeY = (0, 100)

    # Generate a set of all points within 200 of the origin, to be used as offsets later
    # There's probably a more efficient way to do this.
    deltas = set()
    for x in range(-radius, radius+1):
        for y in range(-radius, radius+1):
            if x*x + y*y <= radius*radius:
                deltas.add((x,y))

    randPoints = []
    excluded = set()
    for i in range(0,num_points):
        while i<qty:
            x = random.randrange(*rangeX)
            y = random.randrange(*rangeY)
            if (x,y) in excluded: continue
            randPoints.append((x,y))
            excluded.update((x+dx, y+dy) for (dx,dy) in deltas)

    print (randPoints)


def main():
    generate_points(40);

if __name__=="__main__":
    main()