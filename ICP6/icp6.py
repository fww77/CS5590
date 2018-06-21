from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation as cv
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import pandas
from sklearn.decomposition import PCA

iris = datasets.load_iris()
x = iris.data
y = iris.target
splits = cv.train_test_split(x, y, test_size=0.3)
Train_X, Test_X, Train_Y, Test_Y = splits
NB = GaussianNB()
Pred_Y = NB.fit(Train_X, Train_Y).predict(Test_X)
print ((Test_Y== Pred_Y).sum())
print (len(Test_Y))
print('%.2f%%' % (((Test_Y== Pred_Y).sum()/len(Test_Y))*100))


df = pandas.read_csv('samplestocks.csv')
Y = df[['returns']]
X = df[['dividendyield']]
numberofcluster = range(1, 15) # test K
kmeans = [KMeans(n_clusters=i) for i in numberofcluster]
# variance = [kmeans[i].fit(Y).score(Y) for i in range(len(kmeans))] #percentage of variance for each cluster
# plt.plot(numberofcluster,variance)
# plt.xlabel('Number of Clusters')
# plt.ylabel('variance')
# plt.title('test K')  #get the best cluster number=3
# plt.show() # elbow method
pca = PCA(n_components=1).fit(Y) #Principal Component Analysis
# Convert data that might be overly dispersed into a set of linear combinations
Y = pca.transform(Y)
X = pca.transform(X)
kmeans=KMeans(n_clusters=3)
kmeansoutput=kmeans.fit(Y)
kmeansoutput
plt.figure('K=3')
plt.scatter(X[:, 0], Y[:, 0], c=kmeansoutput.labels_)
plt.xlabel('Dividend Yield')
plt.ylabel('Returns')
plt.title('K=3')
plt.show()






