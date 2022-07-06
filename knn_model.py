# K-Nearest Neighbors (K-NN)

# Importing the libraries
import pandas as pd
import pickle

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X, y)

# Saving model to disk
pickle.dump(classifier, open('knn_model.pkl','wb'))

# Loading model to compare the results
knn_model = pickle.load(open('knn_model.pkl','rb'))
pred = knn_model.predict([[20, 80000]])
def int_to_word(decision):
    word_dict = {0:"Not buy", 1:"Buy"}
    return word_dict[decision]
print(int_to_word(pred[0]))