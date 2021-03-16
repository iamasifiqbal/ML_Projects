import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score,f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix
import math

# Inset the dataset
dataset = pd.read_csv('G:\Dataset\\data_banknote_authentication.csv')
#print(dataset.head)


# Choose specific cloumns for X and y
X = dataset[['Variance', 'Skewness','Cutosis', 'Entropy']] .values
y = dataset['Class'].values

#print(X)
#print(y)

#Pre-process the data and convert the data intor float number

X = StandardScaler().fit(X).transform(X.astype(float))

# Split the dataset into Train set and Test set
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
#print ('Train set:', X_train.shape,  y_train.shape)
#print ('Test set:', X_test.shape,  y_test.shape)

# Implement KNN algorithm with 4
k = 4
neighbors = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
yhat = neighbors.predict(X_test)

#plot confusion matrix

cm = confusion_matrix(y_test, yhat)
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]

TPR = TP/(TP+FN)
TNR = TN/(TN+FP)
Precision = TP/(TP+FP)
Fscore = 2*TP/(2*TP+FP+FN)

print('True Negative Rate:', TNR)
print('True Positive Rate:', TPR)
print('Precision:', Precision)
print('F-score:', Fscore)


# Evaluationg the Performace
print("Train set Accuracy: ", accuracy_score(y_train, neighbors.predict(X_train)))
print("Test set Accuracy: ", accuracy_score(y_test, yhat))
