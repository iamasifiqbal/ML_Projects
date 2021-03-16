import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # For plotting the figure 
from sklearn.preprocessing import StandardScaler # To Fit the data
from sklearn.cluster import KMeans # For k means implementation
from mpl_toolkits.mplot3d import Axes3D # For 3D plotting

# Read the dataset
dataset = pd.read_csv("us-500.csv")
print(dataset.head)
print(dataset.shape)

# Fitting the Dataset
X = dataset.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
# print(Clus_dataSet)

# K means algorithms implementation
clusterNum = 2
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
# print(labels)

# 3D plotting
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
ax.set_xlabel('Alcohol')
ax.set_ylabel('Malic Acid')
ax.set_zlabel('Ash')

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c= labels.astype(np.float))

# Showing the 3D plot
plt.show()