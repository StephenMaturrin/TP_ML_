import numpy as np
np.set_printoptions(threshold=np.nan)
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from  sklearn.decomposition import  PCA
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

import matplotlib
warnings.filterwarnings('ignore')

data = pd.read_csv('./crime.csv', sep=';')
X = data.ix[:, 1:8].values
labels = data.ix[:, 0].values

scaler = preprocessing.StandardScaler().fit(X)

X_scaled = preprocessing.scale(X)

# print( X_scaled)


pca = PCA(n_components=2)

xr_1 = pca.fit(X_scaled).transform(X_scaled)

# print(xr_1)
PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)


print( "variance ")
print(pca.explained_variance_ratio_)

print( " singular_values_ ")
print(pca.singular_values_)

print("Covariance")
print(pca.get_covariance())

print("precision")
print(pca.get_precision())


# lamda * v


plt.scatter(xr_1[:, 0], xr_1[:, 1])
for label, x, y in zip(labels, xr_1[:, 0], xr_1[:, 1]):

    plt.annotate(label, xy=(x, y), xytext=(-0.2, 0.2), textcoords='offset points')
plt.show()
