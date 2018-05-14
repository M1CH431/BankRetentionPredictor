# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dropout


##########
# Pre-pocessing data
#########

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

#   Gets the values indexed from 4-12, when specifying
#   lower / upper bounds, the ends are not included
X = dataset.iloc[:, 3:13].values

#   Gets the exit value aka the value at the 13th index
y = dataset.iloc[:, 13].values

# Encoding categorical data (turn strings into numbers)
#   Need to encode country of origin and gender

# Encoding the country
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

# Encoding the gender
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

#   Making the dummy variable for country since there are more
#   than two countries,  
onehotencoder = OneHotEncoder(categorical_features = [1])

# Fixing X layout
X = onehotencoder.fit_transform(X).toarray()

#   Getting rid of index 0 to avoid dummy var trap, first two values in row correspond to country
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
# This data was taken from a bank that already knew who had left / not left
# the test set is used to see how accurate your ANN is.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling, normalizing
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#############
#   Building the ANN
#############

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer with dropout (randomly disabling a neuron to prevent overfitting(only works well on one data set), stay below 0.5)
# 6 nodes, uniform(close to 0) weights, activation function is rectifier, 11 inputs (11 indepenent vars)
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dropout(p = 0.1))

# Adding the second hidden layer
# ANN knows to expect 11 input vars
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.1))

# Adding the output layer
# want 1 output node to determine y/n, sigmoid function is ideal for output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
# adam is stochastic gradient descent
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Convert to True/False
y_pred = (y_pred > 0.5)

# make and scale 
new_pred = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_pred = (new_pred > 0.5)

# Making the Confusion Matrix to validate your model
cm = confusion_matrix(y_test, y_pred)

# Evaluating the ANN
def build_classifier():
    
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# Building ANN using KerasClassifier object
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100, n_jobs = -1)
# Let the cross-fold begin, TAKE A HIKE, gives you the accuracies of 10 tests with 100 epochs each
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

# Getting average accuracy and standard deviation
mean = accuracies.mean()
variance = accuracies.std()

# Tunning the ANN
def build_classifier(optimizer):
    
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# Building ANN using KerasClassifier object
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)

# Tunning --- optimizer is how it executes (stochastic gradient descent)
parameters = {'batch_size' : [25, 32], 
              'nb_epoch' : [100, 500],
              'optimizer' : ['adam', 'rmsprop']}

# Running lots of tests to get best params
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10)

grid_search = grid_search.fit(X_train, y_train)
best_paramters = grid_search.best_params_
best_accuracy = grid_search.best_score_




