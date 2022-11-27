import copy
import statistics
import random
import math
import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn.cluster import Birch


# Randomly choose k initial centers
def initializeCenters(matrix, k):
    numberCenters = 0
    initialCenters = []
    while numberCenters != k:
        pointIndex = random.randint(0, len(matrix)-1)
        if matrix[pointIndex] not in initialCenters:
            initialCenters.append(matrix[pointIndex])
            numberCenters += 1
    return initialCenters


# Euclidean distances between a point and all centers
# dataPoint is a list, center is a list of centers/lists
def distance(dataPoint, center):
    distanceVec = []
    for points in center:
        distance = math.sqrt(
            sum((dataPoint[values]-points[values])**2 for values in range(len(dataPoint))))
        distanceVec.append(distance)
    return distanceVec


# Update to centers of gravity
# input parameter is a list of assignments, a list of centers, and a matrix
def updateCenter(grouping_result, center, matrix):
    for each_center in center:
        cluster_number = center.index(each_center)
        members = []
        for y in range(len(grouping_result)):
            if grouping_result[y] == cluster_number:
                members.append(matrix[y])

        for column_number in range(len(members[0])):
            column = []
            for z in members:
                column.append(z[column_number])
            mean = statistics.mean(column)
            center[center.index(each_center)][column_number] = mean


# Test for convergence
# both inputs are lists of centers
def convergence(oldCenter, newCenter):
    convergence = False
    for old_center in oldCenter:
        center_index = oldCenter.index(old_center)
        if math.sqrt(sum((old_center[coord]-newCenter[center_index][coord])**2 for coord in range(len(old_center)))) <= (10 ** (-6)):
            convergence = True
        else:
            return False

    return convergence


# k-mean clustering implementation
def kmeanClustering(Matrix, k):

    centers = initializeCenters(Matrix, k)  # a list of centers
    print(f"The initial centers are {centers}")
    iteration = 0
    final_centers = []  # value to return
    cluster_result = []  # value to return

    while True:
        grouping = []  # the assignment to clusters

        # make a copy of centers from last iteration for comparison
        previousCenters = copy.deepcopy(centers)

        # assign each data point to its closest center
        for x in Matrix:
            count = 0
            minLst = []
            distanceVector = distance(x, centers)

            # if a point has equal distance to more than one centers break tie randomly
            for b in distanceVector:
                if b == min(distanceVector):
                    minLst.append(distanceVector.index(b))
                    count += 1
            if count > 1:
                assign_result = random.choice(minLst)
            else:
                assign_result = distanceVector.index(min(distanceVector))

            grouping.append(assign_result)

        # update centers based on the assignment
        updateCenter(grouping, centers, Matrix)

        # check for convergence
        if (convergence(previousCenters, centers)):
            break

        iteration += 1

    final_centers = centers
    print(f"The number of iterations is {iteration}.")
    print(f"The final centers are {final_centers}")

    for a in range(0, k):
        temp_list = []
        for numbers in range(0, len(grouping)):
            if grouping[numbers] == a:
                temp_list.append(Matrix[numbers])
        print(f"There are {len(temp_list)} data points in cluster{(a+1)}.")
        cluster_result.append(temp_list)

    # return a list of centers, and a list of lists/points
    # each list within cluster_result -> points assigned to the same cluster
    return final_centers, cluster_result


# distance between one point and its cluster center
def distance2(point1, point2):

    distance = math.sqrt(
        sum((point1[i]-point2[i])**2 for i in range(len(point1)))) ** 2

    return distance


# return outlier mean in a cluster
def cluster_outlier_mean(center, points):

    vector = np.zeros(len(points))
    mean_outlier = []
    outliers = []
    average = 0

    for j in range(len(points)):
        vector[j] = distance2(center, points[j])

    sorted_vector = np.sort(vector)

    # IQR
    q3, q1 = np.percentile(sorted_vector, [75, 25])
    iqr = q3 - q1

    # outlier
    lower_bound = q1 - 1.5*iqr
    upper_bound = q3 + 1.5*iqr

    for d in range(0, len(vector)):
        if vector[d] < lower_bound:
            outliers.append(points[d])
        elif vector[d] > upper_bound:
            outliers.append(points[d])

    if len(outliers) == 0:
        return "no outliers"
    elif len(outliers) > 1:
        for x in range(0, len(outliers[0])):
            temp_column = []
            for y in range(0, len(outliers)):
                temp_column.append(outliers[y][x])
            average = statistics.mean(temp_column)
            mean_outlier.append(average)
    else:
        return outliers

    return mean_outlier


yeast = pandas.read_csv("yeast.tsv", sep='\t')
yeast = yeast.dropna(axis=0, how='all')
yeast_matrix = yeast.to_numpy()

# extract each column as a separate vector, total 7 columns
for timePoint in range(len(yeast_matrix[0])):
    columnForImpute = []
    naValueIndex = []
    index_count = 0
    for expression in yeast_matrix:
        if np.isnan(expression[timePoint]) == True:
            naValueIndex.append(index_count)
        else:
            columnForImpute.append(expression[timePoint])
        index_count += 1

    meanValue = statistics.mean(columnForImpute)  # mean value for a timepoint

    for nan in naValueIndex:  # update NA values to expression mean at that timepoint
        yeast_matrix[nan][timePoint] = round(meanValue, 3)
# Now we have an expression matrix with no NA value.


# K=3 was implemented as an example, the same was done for other k values
yeast_matrix = yeast_matrix.tolist()
final_centers3, clustering_result3 = kmeanClustering(yeast_matrix, 3)

# plotting
x = ["0hr", "9.5hr", "11.5hr", "13.5", "15.5hr", "18.5hr", "20.5hr"]
y1 = final_centers3[0]
y2 = final_centers3[1]
y3 = final_centers3[2]


plt.figure(1)
plt.plot(x, y1, label="cluster1", marker="o")
plt.plot(x, y2, label="cluster2", marker="o")
plt.plot(x, y3, label="cluster3", marker="o")
plt.xlabel('Time pointes')
plt.ylabel('Expression')
plt.title('Expression changes over time (k=3)')
plt.legend()

outliers1 = cluster_outlier_mean(final_centers3[0], clustering_result3[0])
outliers2 = cluster_outlier_mean(final_centers3[1], clustering_result3[1])
outliers3 = cluster_outlier_mean(final_centers3[2], clustering_result3[2])

plt.figure(2)
plt.plot(x, outliers1, label="outlier", marker="o", linestyle="dashed")
plt.plot(x, y1, label="cluster1 mean", marker="o")
plt.title('Outliers in cluster1')
plt.xlabel('Time pointes')
plt.ylabel('Expression')
plt.legend()


plt.figure(3)
plt.plot(x, outliers2, label="outlier", marker="o", linestyle="dashed")
plt.plot(x, y2, label="cluster2 mean", marker="o")
plt.title('Outliers in cluster2')
plt.xlabel('Time pointes')
plt.ylabel('Expression')
plt.legend()


plt.figure(4)
plt.plot(x, outliers3, label="outlier", marker="o", linestyle="dashed")
plt.plot(x, y3, label="cluster3 mean", marker="o")
plt.title('Outliers in cluster3')
plt.xlabel('Time pointes')
plt.ylabel('Expression')
plt.legend()

plt.show()


# Using another clustering method: Birch
def birch_cluster_mean(matrix, k):

    clusterMean = []  # list of k cluster means
    matrix = np.array(matrix)

    # define the model
    model = Birch(branching_factor=50, threshold=1.5, n_clusters=k)
    # fit the model
    model.fit(matrix)
    # assign a cluster to each example
    prediction = model.predict(matrix)

    # identify cluster centers
    for i in range(0, k):
        assign = []
        for pred in range(0, len(prediction)):
            if prediction[pred] == i:
                assign.append(matrix[pred])

        center = []
        for x in range(0, len(matrix[0])):
            temp_column = []
            for y in range(0, len(assign)):
                temp_column.append(assign[y][x])
            average = statistics.mean(temp_column)
            center.append(average)

        clusterMean.append(center)

    print(f"The final cluster centers by BIRCH are {clusterMean}")
    return clusterMean


birch_centers3 = birch_cluster_mean(yeast_matrix, 3)

# plotting
y4 = birch_centers3[0]
y5 = birch_centers3[1]
y6 = birch_centers3[2]


plt.figure(5)
plt.plot(x, y1, label="K-mean", marker="o", linestyle="-.")
plt.plot(x, y2, label="K-mean", marker="o", linestyle="-.")
plt.plot(x, y3, label="K-mean", marker="o", linestyle="-.")
plt.plot(x, y4, label="K-mean", marker="o", linestyle="-.")
plt.plot(x, y5, label="Birch", marker="o")
plt.plot(x, y6, label="Birch", marker="o")
plt.xlabel('Time pointes')
plt.ylabel('Expression')
plt.title('Cluster centers BIRCH vs. K-mean (k=3)')
plt.legend()

plt.show()
