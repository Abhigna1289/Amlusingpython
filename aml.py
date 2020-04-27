# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 14:14:44 2020

@author: MYPC
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense


#Importing the dataset as a dataframe

df =  pd.read_csv("amlds.csv")
df.head()
df.columns

#removing the unwanted columns

df.drop('nameOrig', axis=1, inplace=True)
df.drop('nameDest', axis=1, inplace=True)
df.drop('isFlaggedFraud', axis=1, inplace=True)

#Checking for any null values

print('Null Values =',df.isnull().values.any())

#correlation matrix to check multicolinearity between the variables

correlation = df.corr()
plt.figure(figsize=(15,15))
plt.title('Correlation Matrix')
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')

#creating dummy variables for categorical values and dropping the original

dum = pd.get_dummies(df['type'])
pf = pd.concat([df,dum],axis=1)
pf.drop(['type'],axis=1, inplace=True)



bf = pf.sample(n=20000)
bf.isFraud.value_counts().plot.bar()
print(bf.isFraud.value_counts())

#Splitting the data into training and test 

X_train, X_test, y_train, y_test = train_test_split(bf.drop(['isFraud'],axis=1), bf['isFraud'], test_size=0.3, random_state=0)

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

# resampling the training data

sm = SMOTE(random_state=10)
x_train_res, y_train_res = sm.fit_sample(X_train, y_train)

print('After OverSampling, the shape of train_X: {}'.format(x_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))

# Feature scaling 


sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train_res)
x_test_scaled = sc.transform(X_test)

''' Initializing the model '''

model = Sequential()

''' Adding the input layer and the first hidden layer '''

model.add(Dense(input_dim=11, output_dim = 6, init = 'uniform', activation = 'relu'))
model.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
model.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

''' Compiling and fitting the model '''

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model_info = model.fit(x_train_scaled, y_train_res, batch_size = 10, nb_epoch = 10)

'''
history = model.fit(x_train_scaled, y_train_res, validation_split=0.2, epochs=10, verbose=1)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
'''


# Predicting the test results

y_pred = model.predict_classes(x_test_scaled)
acc = accuracy_score(y_test,y_pred)*100
print('Accuracy:',round(acc,2))

print("counts of label '1': {}".format(sum(y_pred==1)))
print("counts of label '0': {} \n".format(sum(y_pred==0)))

# Generating the Confusion matrix and Classification report



mat=confusion_matrix(y_test, y_pred)
print('Confusion matrix', '\n', mat, '\n')

plt.figure(figsize=(12, 12))
sns.heatmap(mat, annot=True, fmt="d");
plt.title("Confusion matrix")

print('Classification report', '\n', classification_report(y_test, y_pred), '\n')
