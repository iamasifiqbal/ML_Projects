import numpy as numpy
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # For Decission Tree
from sklearn import preprocessing  # Pre-processing the Dataset
from sklearn.model_selection import train_test_split  # Split the dataset  randomly
from sklearn import metrics  # For Evaluation the Performance

# Attach the dataset
dataset = pd.read_csv("G:\dataset\drug200.csv")
print(dataset.head)

# Choose specific column
X = dataset[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values

# Pre process the dataset
pro_sex = preprocessing.LabelEncoder()
pro_sex.fit(['F','M'])
X[:,1] = pro_sex.transform(X[:,1]) 

pro_BP = preprocessing.LabelEncoder()
pro_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = pro_BP.transform(X[:,2])


pro_Chol = preprocessing.LabelEncoder()
pro_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = pro_Chol.transform(X[:,3]) 

# Prediction datatset
y = dataset["Drug"]


# Split the dataset into Train set and Test set randomly
X_trainset,X_testset,y_trainset, y_testset = train_test_split(X , y ,test_size=0.5, random_state=3)

# Decission Tree implementation
drug_decission_tree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drug_decission_tree.fit(X_trainset, y_trainset)

#Predict the y value 
prediction_tree = drug_decission_tree.predict(X_testset)

# Compare the datset and find the accuracy
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, prediction_tree))



