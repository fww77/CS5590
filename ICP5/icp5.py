
#Question1

# import numpy as np
# import matplotlib.pyplot as plt
# X = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# Y = [1, 3, 2, 5, 7, 8, 8, 9, 10, 12]
# upper = 0
# down = 0
# Pred_Y = []
# Mean_X = np.mean(X)
# Mean_Y = np.mean(Y)
# for i in range(len(X)):
#     K = (X[i]-Mean_Y)*(Y[i]-Mean_Y)
#     G = np.square(X[i]-Mean_X)
#     upper = K + upper
#     down = G + down
# b1 = upper/down   #slope
# b0 = Mean_Y- b1*Mean_X  # Estimating intercept
# Pred_Y = b0+np.multiply(b1,X)
# plt.scatter(X, Y, color='black')  # plotting the initial datapoints
# plt.plot(X, Pred_Y, color='red', linewidth=2)
# plt.show()



#Question2
# import numpy as np
# import matplotlib.pyplot as plt
# import random
# import sklearn.cluster as Kmeans
#
# def create_cluster(X, centroid_pts):
#     cluster = {}
#     for x in X:
#         value = min([(i[0],np.linalg.norm(x - centroid_pts[i[0]]))for i in enumerate(centroid_pts)], key=lambda s:s[1])[0]
#         try:
#             cluster[value].append(x)
#         except:
#             cluster[value] = [x]
#     return cluster
#
#
# def calculate_new_center(cluster):
#     keys =sorted(cluster.keys())
#     newmu = np.array([(np.mean(cluster[k],axis = 0))for k in keys])
#     return newmu
#
# def matched(new_centroids, old_centroids):
#     return (set([tuple(a)for a in new_centroids]) == set([tuple(a)for a in old_centroids]))
#
# def Apply_Kmeans(X, K, N):
#     # selecting random centroids from dataset and by number of clusters.
#     old_centroids = np.random.randint(N, size = K)
#     old_centroid_pts = np.array([X[i]for i in old_centroids])
#
#     cluster_info = create_cluster(X, old_centroid_pts)
#
#     new_centroid_pts=calculate_new_center(cluster_info)
#     itr = 0
#     print("Graph after selecting initial clusters with initial centroids:")
#     plot_cluster(old_centroid_pts,cluster_info,itr)
#     while not matched(new_centroid_pts, old_centroid_pts):
#         itr = itr + 1
#         old_centroid_pts = new_centroid_pts
#         cluster_info = create_cluster(X,new_centroid_pts)
#         plot_cluster(new_centroid_pts, cluster_info,itr)
#         new_centroid_pts = calculate_new_center(cluster_info)
#
#     print("Results after final iteration:")
#     plot_cluster(new_centroid_pts, cluster_info, itr)
#     return
#
# def plot_cluster(mu,cluster, itr):
#     color = 10 * ['r.','g.','k.','c.','b.','m.']
#     print('Iteration number : ',itr)
#     for l in cluster.keys():
#         for m in range(len(cluster[l])):
#             plt.plot(cluster[l][m][0], cluster[l][m][1], color[l], markersize=10)
#     plt.scatter(mu[:,0],mu[:,1],marker = 'x', s = 150, linewidths = 5, zorder = 10)
#     plt.show()
#
# def init_graph(N, p1, p2):
#     X = np.array([(random.uniform(p1,p2),random.uniform(p1,p2))for i in range(N)])
#     return X
#
#
# #
# def Simulate_Clusters():
#     N = int(input('Enter the number of T-shirts;'))
#     K = int(input('Enter the number of Clusters:'))
#     p1 = int(input('Enter the lower bound for height.......'))
#     p2 = int(input('Enter the upper bound for weights.......'))
#     X = init_graph(N, p1, p2)
#     plt.scatter(X[:, 0], X[:, 1])
#     plt.show()
#     temp = Apply_Kmeans(X, K, N)

#
# if __name__ == '__main__':
#     Simulate_Clusters()

import random as rand
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans

def Array(lowHeight: int, highHeight: int, lowWeight: int, highWeight: int, sample: int):
    data = []
    for _ in range(0, sample):
        height = rand.uniform(lowHeight, highHeight)
        chest = rand.uniform(lowWeight, highWeight)
        data.append([height, chest])
    return np.asarray(data)

# generates the random data
X = Array(50, 100, 100, 300, 150) # heights are 50inches to 100inches and weights in range( 100 lbs to 300lbs ) with 150 samples
print(X)

kmeans = KMeans(n_clusters=3) # number of clusters
y_kmeans=kmeans.fit(X) # predict cluster index for each sample.
y_kmeans = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1],c=y_kmeans)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=50); # centroid is red and zise is 50
plt.show()
