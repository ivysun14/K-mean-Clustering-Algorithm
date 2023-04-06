# K-mean-Clustering-Algorithm
K-mean Clustering Implementation

This repository contains the python code for a K-mean clustering algorithm designed and implemented from lists in Python. The algorithm is applied to cluster 6,400 yeast gene expression data collected over 7 timepoints over the course of yeast alcohol fermentation. The implementation includes functions for:

- center initialization for inputted center number
- euclidean distance calculation between a data point to all current cluster centers
- cluster assignment
- center of gravity calculation/center update
- k-mean convergence testing
- outlier detection in each final cluster

The input to the clustering function should be a data matrix, and the desired cluster number k needed to be inputted as well.

The same data set is then used to train a BIRCH clustering model using the Scikit-learn library. Between the two methods, a comparison on their respective cluster means under different k values is made. The project was completed in the class CS CM121 Introduction to Bioinformatics at UCLA.
