# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 22:43:02 2022

@author: Benjamin
"""


# SDA_2022 Assignment 07 Q2
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
np.random.seed(123)


#%% A
pca = PCA()
pca.fit(lfp)
X = pca.transform(lfp)

plt.figure()
plt.plot(pca.components_[0], label='1st comp')
plt.plot(pca.components_[1], label='2nd comp')
plt.plot(pca.components_[2], label='3rd comp')
plt.title('Coefficients of top 3 PCA components - using sklearn')
plt.legend()


#%% B
df = lfp
df_meaned = df - np.mean(df , axis = 0)
cov_mat = np.cov(df_meaned , rowvar = False)
eigen_vals, eigen_vecs = np.linalg.eigh(cov_mat)
inds = np.argsort(eigen_vals)[::-1]
sorted_eigenvals = eigen_vals[inds]
sorted_eigenvecs = eigen_vecs[:,inds]

plt.figure()
plt.plot(sorted_eigenvecs[0], label='1st comp')
plt.plot(sorted_eigenvecs[1], label='2nd comp')
plt.plot(sorted_eigenvecs[2], label='3rd comp')
plt.title('Coefficients of top 3 PCA components - my code')
plt.legend()

eigenvector_subset = sorted_eigenvecs[:,0:3]
A = np.dot(eigenvector_subset.transpose(),df_meaned.transpose()).transpose()

plt.figure()
plt.scatter(A[:,0], A[:,1])
plt.title('Scatterplot of projection of data on PCA components')
plt.xlabel('1st comp')
plt.ylabel('2nd comp')

plt.figure()
plt.scatter(A[:,1], A[:,2])
plt.title('Scatterplot of projection of data on PCA components')
plt.xlabel('2nd comp')
plt.ylabel('3rd comp')

plt.figure()
plt.scatter(A[:,0], A[:,2])
plt.title('Scatterplot of projection of data on PCA components')
plt.xlabel('1st comp')
plt.ylabel('3rd comp')


#%% C
plt.figure()
plt.hist(X[:,0], label='1st comp', alpha=0.7)
plt.hist(X[:,1], label='2nd comp', alpha = 0.7)
plt.hist(X[:,2], label='3rd comp', alpha = 0.7)
plt.title('Histogram of projection of data on PCA components')
plt.legend()

plt.figure()
plt.scatter(X[:,0], X[:,1])
plt.title('Scatterplot of projection of data on PCA components')
plt.xlabel('1st comp')
plt.ylabel('2nd comp')

plt.figure()
plt.scatter(X[:,1], X[:,2])
plt.title('Scatterplot of projection of data on PCA components')
plt.xlabel('2nd comp')
plt.ylabel('3rd comp')

plt.figure()
plt.scatter(X[:,0], X[:,2])
plt.title('Scatterplot of projection of data on PCA components')
plt.xlabel('1st comp')
plt.ylabel('3rd comp')


kmeans = KMeans(n_clusters=2, random_state=0).fit(X[:,0:1])
plt.figure()
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_)
plt.title('Scatterplot of projection of data on PCA components')
plt.xlabel('1st comp')
plt.ylabel('2nd comp')
plt.title('K-means clustering using 2 clusters')

kmeans = KMeans(n_clusters=2, random_state=0).fit(X[:,1:2])
plt.figure()
plt.scatter(X[:,1], X[:,2], c=kmeans.labels_)
plt.xlabel('2nd comp')
plt.ylabel('3rd comp')
plt.title('K-means clustering using 2 clusters')

kmeans = KMeans(n_clusters=2, random_state=0).fit(X[:,(0,2)])
plt.figure()
plt.scatter(X[:,0], X[:,2], c=kmeans.labels_)
plt.xlabel('1st comp')
plt.ylabel('3rd comp')
plt.title('K-means clustering using 2 clusters')

#%% D
print(pca.explained_variance_ratio_[:2])
print(np.sum(pca.explained_variance_ratio_[:2]))
print(np.sum(pca.explained_variance_ratio_[2:]))

#%% E
variances = np.var(lfp, axis=0)

plt.figure()
v = lfp[:,13]
pre_dist = plt.hist(v, density=True, alpha=0.7, bins=np.arange(start=np.min(v), stop=np.max(v), step = 1), label='Dim14')
pre_entropy = sp.entropy(pre_dist[0])

a = X[:,0]
pca_dist = plt.hist(a, density=True, alpha=0.7, bins=np.arange(start=np.min(a), stop=np.max(a), step = 1), label='PCA comp1')
pca_entropy = sp.entropy(pca_dist[0])

plt.title('Distributions')
plt.ylabel('p(x)')
plt.legend()
