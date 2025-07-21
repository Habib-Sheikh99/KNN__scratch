import numpy as np  
import pandas as pd      # We only need these libraries as the algorithm is from scratch
#%%

class KNN:
    def __init__(self, n_neighbors:int=3):
        self.n_neighbors=n_neighbors
        
    def euclidean_distance(self, row1, row2):
        return np.sqrt(np.sum((row1 - row2)**2))
    
    def fit(self, X, y):
        self.X_train=X  
        self.y_train=y  # Wierd! But, this will fit KNN to the model
    
    def get_neighbors(self, test_row):
        distances=[]
        for i, train_row in enumerate(self.X_train):
            distance=self.euclidean_distance(train_row, test_row)
            distances.append((i, distance))
        distances.sort(key=lambda tup:tup[1])
        
        neighbor_labels=[self.y_train[distances[i][0]] for i in range(self.n_neighbors)]  # Labels of nearest data-points (neighbors)
        
        
        return neighbor_labels
    
    def predict_class(self, test_row):
        neighbors=self.get_neighbors(test_row)
        from collections import Counter
        prediction=Counter(neighbors).most_common(1)[0][0]
        return prediction
    
    
    def predict(self, X_test):
        self.X_test=np.array(X_test)         
        return np.array([self.predict_class(row)for row in self.X_test])
         
#%%
### TESTING

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
#%%
dataset = load_iris()
X, y = dataset.data, dataset.target
#%%
X, y = pd.DataFrame(X), pd.Series(y)
#%%
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(
    X, y,
    test_size=0.254,
    random_state=42,
    stratify=y
    )

#%%
xtrain = xtrain.to_numpy()
xtest = xtest.to_numpy()
ytrain = ytrain.to_numpy()
ytest = ytest.to_numpy()
#%%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)
#%%
system=KNN()
system.fit(xtrain, ytrain)
prediction=(system.predict(xtest))
#%%
print("Accuracy : ", accuracy_score(ytest, prediction))
print("Accuracy : ", int(accuracy_score(ytest, prediction)*100), '%')

### ----------------------------------- ###

### Test Result ###

################### ACCURACY : 
#                            ~ 
