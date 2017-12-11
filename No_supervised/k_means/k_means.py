import numpy as np
np.set_printoptions(threshold=np.nan)
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from  sklearn.decomposition import  PCA
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib
from sklearn import  cluster




# loading the folder
data = pd.read_csv('./crime.csv', sep=';')
X = data.ix[:, 1:8].values
Y = data.ix[:, 0].values


# creating an object scaler
scaler = StandardScaler()
StandardScaler(copy=True, with_mean=True, with_std=True)
# loading X in scaler
scaler.fit(X)
# tranforming X by standarscaling
X_scaled = preprocessing.scale(X)

# print( X_scaled)

# selecting two components en PCA
pca = PCA(n_components=2)

# Loading X_scalded and applying transformation
X_pca = pca.fit(X_scaled).transform(X_scaled)



#
PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)

# initializing
kmeans = KMeans(n_clusters=4, random_state=0).fit_predict(X_pca)





# lamda * v


colors = ['red','yellow','blue','pink']
plt.scatter(X_pca[:, 0], X_pca[:, 1], c= kmeans , cmap=matplotlib.colors.ListedColormap(colors))
for label, x, y in zip(Y, X_pca[:, 0], X_pca[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(-0.2, 0.2), textcoords='offset points')
plt.show()
