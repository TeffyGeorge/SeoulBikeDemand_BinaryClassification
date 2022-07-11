#!/usr/bin/env python
# coding: utf-8

# In[3]:


# importing the libraries required
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


# In[4]:


# read data from excel
bike_df = pd.read_csv("SeoulBikeData.csv",encoding='cp1252')


# In[5]:


print("Info Details")
bike_df.info()


# In[6]:


bike_df.head()


# In[7]:


#checking missing values
bike_df.isna().sum()
bike_df.isnull().sum()


# In[8]:


#checking duplicate values
print('Duplicate values : ', len(bike_df[bike_df.duplicated()]))


# In[9]:


# Dependent variable is Rented Bike Count. 
# We will change this numerical output to a binary classification 
# Adding a new variable with threshold set 
threshold = bike_df['Rented Bike Count'].quantile(0.75)
threshold


# In[10]:


# We set the threshold to 1065.25, which is the 75th percentile for Rented Bike Count.
# Every field with a bike count above 1065.25 will be labeled 1, 
# and rest will be labeled 0.
# threshold = 1065.25
bike_df["Rented_Bike_Count_Value"] = (bike_df['Rented Bike Count'] > threshold).astype(float)
bike_df.head(5000)


# # DATA PRE PROCESSING

# In[11]:


# Add few variables (split the date to month, year,weekday and day)
bike_df['Date']=pd.to_datetime(bike_df['Date'])
from datetime import datetime
import datetime as dt

bike_df['Year']=bike_df['Date'].dt.year
bike_df['Month']=bike_df['Date'].dt.month
bike_df['Day']=bike_df['Date'].dt.day
bike_df['DayName']=bike_df['Date'].dt.day_name()
bike_df['Weekday'] = bike_df['DayName'].apply(lambda x : 1 if x=='Saturday' or x=='Sunday' else 0 )
bike_df=bike_df.drop(columns=['Date','DayName','Year'],axis=1)


# In[12]:


#Holiday, Functioning Day and Seasons need to have dummy variables  (object DType)
bike_df = pd.get_dummies(bike_df, columns = ['Seasons',	'Holiday',	'Functioning Day'])


# In[13]:


bike_df.head()


# # DECLARE FEATURE VECTOR AND TARGET VARIABLE

# In[14]:


# Split the features in X and Y
X = bike_df.drop(columns=['Rented_Bike_Count_Value','Rented Bike Count'], axis=1)
y = bike_df['Rented_Bike_Count_Value']


# # SPLIT DATA TO TRAINING AND TEST SET

# In[15]:


#Create test and train data
from sklearn.model_selection import train_test_split
#split the data by percentage
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)

#checking the shape of X_train and X_test
print(X_train.shape)
print(X_test.shape)


# # FEATURE SCALING

# In[16]:


cols = X_train.columns


# In[17]:


print("------Data Standardization------")
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[18]:


X_train = pd.DataFrame(X_train, columns=[cols])


# In[19]:


X_test = pd.DataFrame(X_test, columns=[cols])


# In[20]:


X_train.describe()


# # RUN SVM WITH DEFAULT PARAMETERS

# 1. C = 1.0, kernel = rbf, gamma = auto

# In[21]:


# import SVC classifier
from sklearn.svm import SVC
# import metrics to check accuracy
from sklearn.metrics import accuracy_score

# instantiate classifier with default hyperparameters
svclassifier=SVC(random_state = 0) 

# fit classifier to training set
svclassifier.fit(X_train,y_train)

# Making predictions on the test set
y_pred=svclassifier.predict(X_test)
accuracyscore_C1_train = svclassifier.score(X_train, y_train)
accuracyscore_C1_test = svclassifier.score(X_test, y_test)
# Accuracy score
print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# In[24]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# 2. C = 100.0

# In[21]:


# instantiate classifier with default hyperparameters
svclassifier=SVC(kernel = 'rbf',C = 100.0) 

# fit classifier to training set
svclassifier.fit(X_train,y_train)

# Making predictions on the test set
y_pred=svclassifier.predict(X_test)
accuracyscore_C100_train = svclassifier.score(X_train, y_train)
accuracyscore_C100_test = svclassifier.score(X_test, y_test)

# Accuracy score
print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# 3. C = 1000.0

# In[22]:


# instantiate classifier with default hyperparameters
svclassifier=SVC(C = 1000.0) 

# fit classifier to training set
svclassifier.fit(X_train,y_train)

# Making predictions on the test set
y_pred=svclassifier.predict(X_test)
accuracyscore_C1000_train = svclassifier.score(X_train, y_train)
accuracyscore_C1000_test = svclassifier.score(X_test, y_test)
# Accuracy score
print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# We can see that accuracy score is highest for C = 100. Higher the C value, it indicates that the outliers are less

# In[24]:


# Plotting Train and Test Accuracy for C parameter 
C = np.array([1, 100,1000])
train_accuracy = [accuracyscore_C1_train,accuracyscore_C100_train,accuracyscore_C1000_train]
test_accuracy = [accuracyscore_C1_test,accuracyscore_C100_test,accuracyscore_C1000_test]
plt.figure(figsize=(8, 6))
plt.plot(C, train_accuracy)
plt.plot(C, test_accuracy)
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title('rbf Kernel')
plt.legend(['train accuracy','test accuracy'], loc='upper right')
plt.show()


# # SVM LINEAR KERNEL

# 1. Linear Kernel and C = 1.0

# In[37]:


linear_svclassifier=SVC(kernel='linear', C=1.0) 
linear_svclassifier.fit(X_train,y_train)
y_pred_test=linear_svclassifier.predict(X_test)
accuracyscore_C1_train = linear_svclassifier.score(X_train, y_train)
accuracyscore_C1_test = linear_svclassifier.score(X_test, y_test)
# Accuracy score
print('Model accuracy score with linear kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))


# 2. Linear Kernel and C = 100.0

# In[38]:


linear_svclassifier=SVC(kernel='linear', C=100.0) 
linear_svclassifier.fit(X_train,y_train)
y_pred_test=linear_svclassifier.predict(X_test)
accuracyscore_C100_train = linear_svclassifier.score(X_train, y_train)
accuracyscore_C100_test = linear_svclassifier.score(X_test, y_test)
# Accuracy score
print('Model accuracy score with linear kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))


# 3. Linear Kernel and C = 1000.0

# In[39]:


linear_svclassifier=SVC(kernel='linear', C=1000.0) 
linear_svclassifier.fit(X_train,y_train)
y_pred_test=linear_svclassifier.predict(X_test)
accuracyscore_C1000_train = linear_svclassifier.score(X_train, y_train)
accuracyscore_C1000_test = linear_svclassifier.score(X_test, y_test)

# Accuracy score
print('Model accuracy score with linear kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))


# C = 1.0 and C = 1000.0 gives higher accuracy that C = 100.0

# In[ ]:


y_pred_train = linear_svclassifier.predict(X_train)
y_pred_train


# In[ ]:


print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))


# In[72]:


print('Training set score: {:.4f}'.format(linear_svclassifier.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(linear_svclassifier.score(X_test, y_test)))


# The train and test scores show that there is no underfitting or overfitting as the scores are almost same.

# In[28]:


# We will check the Null Accuracy as this is the accuracy that predicts the most frequent class
y_test.value_counts()


# In[29]:


null_accuracy = (1992/(1992+636))

print('Null accuracy score: {0:0.4f}'. format(null_accuracy))


# SVM Accuracy is better than Null Accuracy which shows that SVM classifier is better at classfying the labels

# In[40]:


# Plotting Train and Test Accuracy for C parameter 
C = np.array([1, 100,1000])
train_accuracy = [accuracyscore_C1_train,accuracyscore_C100_train,accuracyscore_C1000_train]
test_accuracy = [accuracyscore_C1_test,accuracyscore_C100_test,accuracyscore_C1000_test]
plt.figure(figsize=(8, 6))
plt.plot(C, train_accuracy)
plt.plot(C, test_accuracy)
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title('Linear Kernel')
plt.legend(['train accuracy','test accuracy'], loc='upper right')
plt.show()


# # SVM Polynomial Kernel 

# 1. Polynomial Kernel and C = 1.0

# In[42]:


poly_svclassifier=SVC(kernel='poly', C=1.0) 
poly_svclassifier.fit(X_train,y_train)
y_pred_poly=poly_svclassifier.predict(X_test)
accuracyscore_C1_train = poly_svclassifier.score(X_train, y_train)
accuracyscore_C1_test = poly_svclassifier.score(X_test, y_test)

print('Model accuracy score with polynomial kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_pred_poly, y_pred)))


# 2. Polynomial Kernel and C = 100.0

# In[43]:


poly_svclassifier=SVC(kernel='poly', C=100.0) 
poly_svclassifier.fit(X_train,y_train)
y_pred_poly=poly_svclassifier.predict(X_test)
accuracyscore_C100_train = poly_svclassifier.score(X_train, y_train)
accuracyscore_C100_test = poly_svclassifier.score(X_test, y_test)
print('Model accuracy score with polynomial kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred_poly)))


# 3. Polynomial Kernel and C = 1000.0

# In[44]:


poly_svclassifier=SVC(kernel='poly', C=1000.0) 
poly_svclassifier.fit(X_train,y_train)
y_pred_poly=poly_svclassifier.predict(X_test)
accuracyscore_C1000_train = poly_svclassifier.score(X_train, y_train)
accuracyscore_C1000_test = poly_svclassifier.score(X_test, y_test)
print('Model accuracy score with polynomial kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred_poly)))


# POLYNOMIAL model gives better result than LINEAR kernel

# In[45]:


C = np.array([1, 100,1000])
train_accuracy = [accuracyscore_C1_train,accuracyscore_C100_train,accuracyscore_C1000_train]
test_accuracy = [accuracyscore_C1_test,accuracyscore_C100_test,accuracyscore_C1000_test]
plt.figure(figsize=(8, 6))
plt.plot(C, train_accuracy)
plt.plot(C, test_accuracy)
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title('Polynomial Kernel')
plt.legend(['train accuracy','test accuracy'], loc='upper right')
plt.show()


# # SVM SIGMOID KERNEL

# 1. Sigmoid Kernel and C = 1.0

# In[46]:


sigmoid_svclassifier=SVC(kernel='sigmoid', C=1.0) 
sigmoid_svclassifier.fit(X_train,y_train)
y_pred=sigmoid_svclassifier.predict(X_test)
accuracyscore_C1_train = sigmoid_svclassifier.score(X_train, y_train)
accuracyscore_C1_test = sigmoid_svclassifier.score(X_test, y_test)
print('Model accuracy score with polynomial kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# 2. Sigmoid Kernel and C = 100.0

# In[47]:


sigmoid_svclassifier=SVC(kernel='sigmoid', C=100.0) 
sigmoid_svclassifier.fit(X_train,y_train)
y_pred=sigmoid_svclassifier.predict(X_test)
accuracyscore_C100_train = sigmoid_svclassifier.score(X_train, y_train)
accuracyscore_C100_test = sigmoid_svclassifier.score(X_test, y_test)
print('Model accuracy score with polynomial kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# 3. Sigmoid Kernel and C = 1000.0

# In[48]:


sigmoid_svclassifier=SVC(kernel='sigmoid', C=1000.0) 
sigmoid_svclassifier.fit(X_train,y_train)
y_pred=sigmoid_svclassifier.predict(X_test)
accuracyscore_C1000_train = sigmoid_svclassifier.score(X_train, y_train)
accuracyscore_C1000_test = sigmoid_svclassifier.score(X_test, y_test)
print('Model accuracy score with polynomial kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# SIGMOID kernel performs the poorest among the kernels

# In[49]:


C = np.array([1, 100,1000])
train_accuracy = [accuracyscore_C1_train,accuracyscore_C100_train,accuracyscore_C1000_train]
test_accuracy = [accuracyscore_C1_test,accuracyscore_C100_test,accuracyscore_C1000_test]
plt.figure(figsize=(8, 6))
plt.plot(C, train_accuracy)
plt.plot(C, test_accuracy)
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title('Sigmoid kernel')
plt.legend(['train accuracy','test accuracy'], loc='upper right')
plt.show()


# In[36]:


# compute ROC AUC

from sklearn.metrics import roc_auc_score
ROC_AUC = roc_auc_score(y_test, y_pred_test)
print('ROC AUC : {:.4f}'.format(ROC_AUC))


# GRAPHICAL PLOTS

# In[19]:


# Plot learning curves
# plt.figure(figsize=(10,6))
# plt.title("Normal kernel with C=0.1", fontsize=18)
# plt.scatter(X_train, c=y_train, s=50, cmap='cool')
# plot_svc_decision_function(svclassifier)
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVC

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# 1. rbf Score Vs Training examples

# In[21]:


title = "Learning Curves (SVM, rbf kernel,C = 100, $\gamma=0.1$)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = SVC(C = 100.0, gamma=0.1)
plot_learning_curve(estimator, title, X, y, cv=cv, n_jobs=4)

plt.show()


# In[22]:


title = "Learning Curves (SVM, rbf kernel,C = 100, $\gamma=0.1$)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = SVC(C = 100.0, gamma=0.01)
plot_learning_curve(estimator, title, X, y, cv=cv, n_jobs=4)

plt.show()


# 2. poly Score Vs Training Examples

# In[ ]:


title = "Learning Curves (SVM, poly kernel,C = 100, $\gamma=0.01$)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
estimator = SVC(kernel = 'poly',C = 100.0, gamma=0.1)
plot_learning_curve(estimator, title, X, y, cv=cv, n_jobs=4)
plt.show()


# In[20]:


#Sigmoid Kernel
title = "Learning Curves (SVM, sigmoid kernel,C = 1, $\gamma=0.1$)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = SVC(kernel = 'sigmoid',C = 1.0, gamma=0.1)
plot_learning_curve(estimator, title, X, y, cv=cv, n_jobs=4) 

plt.show()


# In[20]:


#Linear Kernel
title = "Learning Curves (SVM, linear kernel,C = 1)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
estimator = SVC(kernel = 'linear',C = 1.0)
plot_learning_curve(estimator, title, X, y, cv=cv, n_jobs=4)
plt.show()


# # CROSS VALIDATION - SVM

# In[77]:


def cross_validation (classifier):
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    num_folds = 20
    kfold_validation = KFold(n_splits=num_folds)
    results = cross_val_score(classifier, X, y, cv = kfold_validation)
    accuracy=np.mean(abs(results))
    print('Average accuracy: ',accuracy)
    print('Standard Deviation: ',results.std())


# 1. Cross Validation On SVM Model - Default

# In[73]:


# 1. For SVC rbf Default = C = 100.0 [0.9243]
cross_validation(svclassifier)


# 2. Cross Validation of SVM Model - Poly

# In[74]:


# For SVC Default = C = 100.0 [0.9212]
cross_validation(poly_svclassifier)


# In[75]:


# For Sigmoid at C = 1.0
cross_validation(sigmoid_svclassifier)


# In[76]:


# For linear
cross_validation(linear_svclassifier)


# In[78]:


# For linear for kFold 20
cross_validation(linear_svclassifier)


# In[1]:


# CPU available
import os
n_cpu = os.cpu_count()
n_cpu


# In[ ]:


# CROSS VALIDATION - DEFAULT rbf


# In[ ]:


#Ignore next 2 steps as computation is high and is taking time


# In[ ]:


# Checking Cross Validation for SVM using GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
# Define Function
svc_kernel = SVC(random_state = 0, kernel = 'poly')

#define a list of parameters
param_svc_kernel = {'C':  [ 0.1, 1, 10, 100,1000]     ,
                    'gamma':   [ 0.1,1]   } # C = 10,000 mimics hard-margin SVM

#apply grid search
grid_svc_kernel = GridSearchCV(svc_kernel, param_svc_kernel, cv = 5, n_jobs=2)
grid_svc_kernel.fit(X_train, y_train)


# In[ ]:


from sklearn.model_selection import KFold
folds = KFold(n_splits= 5,shuffle = True, random_state = 4)
param_svc_kernel = [{'C':  [ 1, 10, 100, 1000] ,'gamma':   [ 0.1,1]} ]

# Checking Cross Validation for SVM using GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
# Define Function
svc_kernel = SVC(kernel = "poly")

#apply grid search
grid_svc_kernel = GridSearchCV(estimator = svc_kernel, param_grid = param_svc_kernel, cv= folds, 
                              verbose = 1, return_train_score = True)
grid_svc_kernel.fit(X_train, y_train)


# In[ ]:


grid_svc_kernel.best_params_


# In[ ]:


grid_svc_kernel.score(X_test, y_test)


# In[64]:


# CROSS VALIDATION - rbf 


# In[22]:


# creating a kFold object with 5 splits
from sklearn.model_selection import KFold
folds = KFold(n_splits= 5,shuffle = True, random_state = 4)
param_svc_kernel = [{'C':  [ 1, 10, 100, 1000] } ]

# Checking Cross Validation for SVM using GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
# Define Function
svc_kernel = SVC(kernel = "rbf")

#apply grid search
grid_svc_kernel = GridSearchCV(estimator = svc_kernel, param_grid = param_svc_kernel, cv= folds, 
                              verbose = 1, return_train_score = True)
grid_svc_kernel.fit(X_train, y_train)


# In[23]:


grid_svc_kernel.best_params_


# In[24]:


grid_svc_kernel.score(X_test, y_test)


# In[23]:


# CROSS VALIDATION - Polynomial 


# In[24]:


# creating a kFold object with 5 splits
from sklearn.model_selection import KFold
folds = KFold(n_splits= 5,shuffle = True, random_state = 4)
param_svc_kernel = [{'C':  [ 1, 10, 100, 1000] } ]

# Checking Cross Validation for SVM using GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
# Define Function
svc_kernel = SVC(kernel = "poly")

#apply grid search
grid_svc_kernel = GridSearchCV(estimator = svc_kernel, param_grid = param_svc_kernel, cv= folds, 
                              verbose = 1, return_train_score = True)
grid_svc_kernel.fit(X_train, y_train)


# In[25]:


grid_svc_kernel.best_params_


# In[26]:


grid_svc_kernel.score(X_test, y_test)


# In[27]:


# CROSS VALIDATION - Sigmoid 


# In[28]:


# creating a kFold object with 5 splits
from sklearn.model_selection import KFold
folds = KFold(n_splits= 5,shuffle = True, random_state = 4)
param_svc_kernel = [{'C':  [ 1, 10, 100, 1000] } ]

# Checking Cross Validation for SVM using GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
# Define Function
svc_kernel = SVC(kernel = "sigmoid")

#apply grid search
grid_svc_kernel = GridSearchCV(estimator = svc_kernel, param_grid = param_svc_kernel, cv= folds, 
                              verbose = 1, return_train_score = True)
grid_svc_kernel.fit(X_train, y_train)


# In[29]:


grid_svc_kernel.best_params_


# In[30]:


grid_svc_kernel.score(X_test, y_test)


# In[ ]:


# CROSS VALIDATION - Linear 


# In[59]:


# creating a kFold object with 5 splits
from sklearn.model_selection import KFold
folds = KFold(n_splits= 5,shuffle = True, random_state = 4)
param_svc_kernel = [{'C':  [ 1, 10, 100, 1000] } ]

# Checking Cross Validation for SVM using GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
# Define Function
svc_kernel = SVC(kernel = "linear")

#apply grid search
grid_svc_kernel = GridSearchCV(estimator = svc_kernel, param_grid = param_svc_kernel, cv= folds, 
                              verbose = 1, return_train_score = True)
grid_svc_kernel.fit(X_train, y_train)


# In[60]:


grid_svc_kernel.best_params_


# In[61]:


grid_svc_kernel.score(X_test, y_test)


# # DECISION TREE 

# # 1. Unpruned Tree

# In[31]:


# import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state=0)
classifier.fit(X_train, y_train)
y_train_predict = classifier.predict(X_train)
y_test_predict = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print(f'Training accuracy score - Default Tree :{accuracy_score(y_train_predict, y_train)}')
print(f'Test accuracy score - Default Tree :{accuracy_score(y_test_predict, y_test)}')


# Here, we can see that model is overfitted with unpruned decision tree.The Training and testing score have difference

# # 2. Decision Tree With GINI Index

# In[32]:


classifier_gini = DecisionTreeClassifier(criterion='gini', max_depth=6, random_state=0)
classifier_gini.fit(X_train, y_train)
y_train_predict_gini = classifier_gini.predict(X_train)
y_test_predict_gini = classifier_gini.predict(X_test)

from sklearn.metrics import accuracy_score
print(f'Training accuracy score - Default Tree With GINI :{accuracy_score(y_train_predict_gini, y_train)}')
print(f'Test accuracy score - Default Tree With GINI :{accuracy_score(y_test_predict_gini, y_test)}')


# Check for overfitting and underfitting

# In[33]:


# Scores of training and test 
print('Training set score: {:.4f}'.format(classifier_gini.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(classifier_gini.score(X_test, y_test)))


# The training-set accuracy score is 0.8994 while the test-set accuracy to be 0.9022. These two values are quite comparable. So, there is no sign of overfitting.

# # 3. Decision Tree with Entropy

# In[34]:


classifier_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=0)
classifier_entropy.fit(X_train, y_train)
y_train_predict_entropy = classifier_entropy.predict(X_train)
y_test_predict_entropy = classifier_entropy.predict(X_test)

from sklearn.metrics import accuracy_score
print(f'Training accuracy score - Default Tree With Entropy :{accuracy_score(y_train_predict_entropy, y_train)}')
print(f'Test accuracy score - Default Tree With Entropy :{accuracy_score(y_test_predict_entropy, y_test)}')


# Check for overfitting or underfitting

# In[35]:


# Scores of training and test 
print('Training set score: {:.4f}'.format(classifier_entropy.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(classifier_entropy.score(X_test, y_test)))


# We can see that the training-set score and test-set score is same as above. The training-set accuracy score is 0.9185 while the test-set accuracy to be 0.9189. These two values are quite comparable. So, there is no sign of overfitting.
# 
# Now, based on the above analysis we can conclude that our classification model accuracy is very good. Model will do a good job in prediction.We will now see if there are any errors

# # 4. Decision Tree - Post Pruning

# In[36]:


path = classifier.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
print(ccp_alphas)


# In[37]:


# We will append the model to our list for each alpha
classifiers = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state = 0, ccp_alpha = ccp_alpha)
    clf.fit(X_train,y_train)
    classifiers.append(clf)


# In[38]:


train_data = []
test_data = []
for c in classifiers:
    y_train_predict = c.predict(X_train)
    y_test_predict = c.predict(X_test)
    train_data.append(accuracy_score(y_train_predict,y_train))
    test_data.append(accuracy_score(y_test_predict,y_test))

plt.scatter(ccp_alphas, train_data)
plt.scatter(ccp_alphas, test_data)
plt.plot(ccp_alphas, train_data, label= 'Train Accuracy', drawstyle = 'steps-post')
plt.plot(ccp_alphas, test_data, label= 'Test Accuracy', drawstyle = 'steps-post')
plt.legend()
plt.title('Accuracy Vs Alpha')
plt.show()


# We will choose 0.002 as ccp_alpha as we get the maximum accuracy here. The Train accuracy is also good.

# In[39]:


classifier_postpruning = DecisionTreeClassifier(criterion='entropy', ccp_alpha = 0.002)
classifier_postpruning.fit(X_train, y_train)
y_train_predict_postpruning = classifier_postpruning.predict(X_train)
y_test_predict_postpruning = classifier_postpruning.predict(X_test)

from sklearn.metrics import accuracy_score
print(f'Training accuracy score - Decision Tree - Post Pruning :{accuracy_score(y_train_predict_postpruning, y_train)}')
print(f'Test accuracy score - Decision Tree - Post Pruning :{accuracy_score(y_test_predict_postpruning, y_test)}')


# The accuracy score for default scenario is checked

# In[58]:


classifier_gini = DecisionTreeClassifier(criterion='gini',random_state=0)
classifier_gini.fit(X_train, y_train)
y_train_predict_gini = classifier_gini.predict(X_train)
y_test_predict_gini = classifier_gini.predict(X_test)

from sklearn.metrics import accuracy_score
print(f'Training accuracy score - Default Tree With GINI :{accuracy_score(y_train_predict_gini, y_train)}')
print(f'Test accuracy score - Default Tree With GINI :{accuracy_score(y_test_predict_gini, y_test)}')


# # CROSS VALIDATION - DECISION TREE 

# In[75]:


# Classifier - Unpruned Decision Tree
cross_validation(classifier)


# In[76]:


# Classifier - Decision Tree - GINI
cross_validation(classifier_gini)


# In[77]:


# Classifier - Decision Tree - Entropy
cross_validation(classifier_entropy)


# In[79]:


# Classifier - Decision Tree - Post Pruned
cross_validation(classifier_postpruning)


# CROSS VALIDATION USING GridSearchCV
# 
# 1. Unpruned Tree

# In[41]:


# Unpruned Tree
# GridSearch + CV
from sklearn.model_selection import GridSearchCV

# estimator
opt_tree = DecisionTreeClassifier(random_state = 0) 
##  random state is set for all models (hyperparameter specifications)

# hyperparameter setting
dt_params = {'max_depth':  range(1,10) ,
             'min_samples_split':   range(2,11),
             'max_leaf_nodes':    range(2,11)   }


# gridsearch function
grid_tree = GridSearchCV(opt_tree, dt_params)

grid_tree.fit(X_train, y_train)


# In[42]:


# Report the best hyperparameters chosen
grid_tree.best_params_


# In[43]:


grid_tree.best_score_


# 2. Entropy

# In[45]:


# ENTROPY
# GridSearch + CV
from sklearn.model_selection import GridSearchCV

# estimator
opt_tree = DecisionTreeClassifier(random_state = 0, criterion = 'entropy') 
##  random state is set for all models (hyperparameter specifications)

# hyperparameter setting
dt_params = {'max_depth':  range(1,10) ,
             'min_samples_split':   range(2,11),
             'max_leaf_nodes':    range(2,11)   }


# gridsearch function
grid_tree = GridSearchCV(opt_tree, dt_params)

grid_tree.fit(X_train, y_train)


# In[46]:


# Report the best hyperparameters chosen
grid_tree.best_params_


# In[47]:


grid_tree.best_score_


# 3. GINI

# In[48]:


# ENTROPY
# GridSearch + CV
from sklearn.model_selection import GridSearchCV

# estimator
opt_tree = DecisionTreeClassifier(random_state = 0, criterion = 'gini') 
##  random state is set for all models (hyperparameter specifications)

# hyperparameter setting
dt_params = {'max_depth':  range(1,10) ,
             'min_samples_split':   range(2,11),
             'max_leaf_nodes':    range(2,11)   }


# gridsearch function
grid_tree = GridSearchCV(opt_tree, dt_params)

grid_tree.fit(X_train, y_train)


# In[49]:


# Report the best hyperparameters chosen
grid_tree.best_params_


# In[50]:


grid_tree.best_score_


# 4.POST PRUNING

# In[51]:


# POST PRUNING
# GridSearch + CV
from sklearn.model_selection import GridSearchCV

# estimator
opt_tree = DecisionTreeClassifier(criterion='entropy', ccp_alpha = 0.002) 
##  random state is set for all models (hyperparameter specifications)

# hyperparameter setting
dt_params = {'max_depth':  range(1,10) ,
             'min_samples_split':   range(2,11),
             'max_leaf_nodes':    range(2,11)   }


# gridsearch function
grid_tree = GridSearchCV(opt_tree, dt_params)

grid_tree.fit(X_train, y_train)


# In[52]:


# Report the best hyperparameters chosen
grid_tree.best_params_


# In[91]:


grid_tree.best_score_


# In[100]:


# Post Pruning Tree
from sklearn import tree
# grid_tree.best_estimator_
print(tree.export_text(grid_tree.best_estimator_))


# In[111]:


X_train.head()


# In[85]:


# Plot the decision tree
from sklearn import tree

fig = plt.figure(figsize=(20,10)) # set a proper figure size (in case that the figure is too small to read or ratio is not proper)

tree.plot_tree(grid_tree.best_estimator_, 
               filled = True, impurity = True) # whether to color the boxes, whether to report gini index
             #   fontsize = 12) # set fontsize to read
plt.show()


# # CLASSIFICATION METRICS 

# We will check the classification report for the classifier which performed the best

# 1. Decision Tree - GINI
# 2. Decision Tree - Entropy
# 3. Decision Tree - Post Pruning
# 4. SVM - DEFAULT parameters (0.9243)
# 5. SVM - POLYNOMIAL and C= 100.0 (0.9212)
# 

# In[45]:


#After running Decision Tree - Unpruned
def classification_details (y_test,y_test_predicted):
    confmatrix = confusion_matrix(y_test, y_test_predicted)
    print('Confusion matrix\n\n', confmatrix)
    print('\nTrue Positives(TP) = ', confmatrix[0,0])
    print('\nTrue Negatives(TN) = ', confmatrix[1,1])
    print('\nFalse Positives(FP) = ', confmatrix[0,1])
    print('\nFalse Negatives(FN) = ', confmatrix[1,0])

    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_test_predicted))

    #Classification Accuracy
    TP = confmatrix[0,0]
    TN = confmatrix[1,1]
    FP = confmatrix[0,1]
    FN = confmatrix[1,0]

    classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
    print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))

    # print classification error
    classification_error = (FP + FN) / float(TP + TN + FP + FN)
    print('Classification error : {0:0.4f}'.format(classification_error))

    # print precision score
    precision = TP / float(TP + FP)
    print('Precision : {0:0.4f}'.format(precision))

    recall = TP / float(TP + FN)
    print('Recall or Sensitivity : {0:0.4f}'.format(recall))

    true_positive_rate = TP / float(TP + FN)
    print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))

    false_positive_rate = FP / float(FP + TN)
    print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))

    specificity = TN / (TN + FP)
    print('Specificity : {0:0.4f}'.format(specificity))
    
    # plot ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_test, y_test_predicted)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0,1], [0,1], 'k--' )
    plt.rcParams['font.size'] = 12
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.show()
    
    # compute ROC AUC
    from sklearn.metrics import roc_auc_score
    ROC_AUC = roc_auc_score(y_test, y_test_predicted)
    print('ROC AUC : {:.4f}'.format(ROC_AUC))


# In[50]:


#1 Decision Tree - GINI
classification_details(y_test, y_test_predict_gini)


# In[49]:


#2 Decision Tree - Entropy
classification_details(y_test, y_test_predict_entropy)


# In[85]:


#3 Decision Tree - Post Pruned
classification_details(y_test, y_test_predict_postpruning)


# In[48]:


#4 SVM - Default at C = 100.0
classification_details(y_test, y_pred)


# In[51]:


#4 SVM - Default at Poly, C = 100.0
classification_details(y_test, y_pred_poly)

