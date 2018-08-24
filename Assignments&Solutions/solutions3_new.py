#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 09:48:04 2017

@author: ltostrams
"""
#%% 3.1.1
import numpy as np
from scipy.io import loadmat

wine_data = loadmat('Data/wine.mat')

X = np.array(wine_data['X'])
y = np.array(wine_data['y'])

y = y.ravel()

attributeNames = [i[0] for i in wine_data['attributeNames'][0]]
classNames = [j[0] for i in wine_data['classNames'] for j in i]

#%%3.1.2
from sklearn import tree
import treeprint

dtc = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=100, max_depth = 18)
dtc = dtc.fit(X,y)


treeprint.tree_print(dtc, attributeNames, classNames)

"""
The min_samples_split parameter sets the minimum number of samples that need to be at an 
internal node to split it. Lowering the value will therefore tend to increase the number of splits
and thus the size of the tree.
"""


#%% 3.1.3
x = np.mat([6.9, 1.09, .06, 2.1, .0061, 12, 31, .99, 3.5, .64, 12])
x_class = classNames[dtc.predict(x)[0]]   # 'white'

"""
As seen in the plot of the tree from 3.1.2, only Total sulfer dioxide (var. 7), 
Chlorides (var. 5) and Sulphates (var. 10) are used for classifying this wine.
"""

#%% 3.1.4
pred = dtc.predict(X)
accuracy = np.mean(pred == y) # 98.68%

#%% 3.2.1
from sklearn import model_selection, tree
import matplotlib.pyplot as plt

test_proportion = 0.33
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=test_proportion)
levels = range(2,21)
error = np.zeros((2,len(levels)))

for t in levels:
    dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
    dtc = dtc.fit(X_train,y_train)
    
    y_est_test = dtc.predict(X_test)
    y_est_train = dtc.predict(X_train)
    
    train_class_error = 1-np.mean(y_est_test == y_test)
    test_class_error = 1-np.mean(y_est_train == y_train)
    error[0,t-2], error[1,t-2]= train_class_error, test_class_error

plt.plot(levels, error[0,:])
plt.plot(levels, error[1,:])
plt.xlabel('Model complexity (max tree depth)')
plt.ylabel('Error (misclassification rate)')
plt.legend(['Error_train','Error_test'])
    
plt.show()  


"""
A very common mistake with the error plots is to plot the accuracies instead of the error rates.
Check the actual values: if the 'error rate' on the training data is above 0.5 (for two classes) 
you very likely did something wrong, since even with no attributes at all a predictor should be 
correct at least half the time; eg. for the wine data, always predicting 'White' should get an
accuracy of about 75%, since that is the proportion of white wines in the data (and thus, on average,
the proportion of white wines in the training data as well).

Some rules of thumb for the plot: training error should always stay the same or decrease as the tree
gets more complex (ie. the max_depth increases), since it can then better fit the data. Training error should 
also be lower than test error (though on occasion, especially with small samples, this might not be 
true in some cases).

The optimal pruning level is typically the level with the lowest test error, though this value can vary somewhat
if you split the data a different way. The point of techniques like K-fold cross-validation is to
take that variation between splits into account.
"""


#%% 3.2.2
K = 10
CV = model_selection.KFold(K,shuffle=True)

error_train = np.zeros((len(levels),10))
error_test = np.zeros((len(levels),10))
k=0
for train_index, test_index in CV.split(X):
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]
    
    for t in levels:
        dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
        dtc = dtc.fit(X_train,y_train)
        
        y_est_test = dtc.predict(X_test)
        y_est_train = dtc.predict(X_train)
        
        train_class_error = 1-np.mean(y_est_test == y_test)
        test_class_error = 1-np.mean(y_est_train == y_train)
        error_train[t-2,k], error_test[t-2,k]= train_class_error, test_class_error
    k=k+1


plt.plot(levels, error_train.mean(1))
plt.plot(levels, error_test.mean(1))
plt.xlabel('Model complexity (max tree depth)')
plt.ylabel('Error (misclassification rate, CV K={0})'.format(K))
plt.legend(['Error_train','Error_test'])
    
plt.show()

"""
Results here should be similar to 3.2.1, but the error rates will tend to be more stable when you 
repeat with a new random split.
"""



#%% 3.3.1
import xlrd

doc = xlrd.open_workbook('Data/classprobs.xls').sheet_by_index(0)
true_class = np.array(doc.col_values(0))
pred1 = np.array(doc.col_values(1))
pred2 = np.array(doc.col_values(2))

#%% 3.3.2
from sklearn.metrics import roc_curve

FP1, TP1, thresh1 = roc_curve(true_class, pred1)
FP2, TP2, thresh2 = roc_curve(true_class, pred2)

plt.plot(FP1, TP1, label='Classifier 1')
plt.plot(FP2, TP2, label='Classifier 2')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")
plt.show()

"""
The diagonal line reflects random guessing. Since both ROC curves are fully above this diagonal
we can conclude that both predictors perform better than random guessing. Moreover, since the curve 
for classifier 1 lies fully above that of classifier 2, we can also conclude that classifier 1 has
the best performance of the two. 
"""


#%% 3.3.3
TC1 = true_class==1
TC0 = true_class==0

auc1 = np.mean([np.mean(pred1[TC1] > i) for i in pred1[TC0]])  # 0.956
auc2 = np.mean([np.mean(pred2[TC1] > i) for i in pred2[TC0]])  # 0.765

"""
The random guessing baseline has an AUC of 0.5, so we can conclude that both classifiers perform better.
The AUC scores also confirm the earlier conclusion that classifier 1 performs better than
classifier 2.
"""


#%% 3.3.4

class1 = np.mean((pred1>0.5) == true_class) # 0.861
class2 = np.mean((pred2>0.5) == true_class) # 0.694

#%% 3.3.5

#really don't think this HAS to be put in a table as long as the numbers are right
table = np.zeros([2,2])

#Number of times both 1 and 2 are right
N12c = sum( ((pred2>0.5) == true_class) & ((pred1>0.5) == true_class)) 

#Number of times both 1 and 2 are wrong
N12w = sum( ((pred2>0.5) != true_class) & ((pred1>0.5) != true_class)) 

#Number of times 2 is right and 1 is wrong
N1s2 = sum( ((pred2>0.5) == true_class) & ((pred1>0.5) != true_class)) 

#Number of times 1 is right and 2 is wrong
N2s1 = sum( ((pred2>0.5) != true_class) & ((pred1>0.5) == true_class))

table[0,0] = N12c #67
table[0,1] = N2s1 #26
table[1,0] = N1s2 #8
table[1,1] = N12w #7

#total number of differences
N = N1s2 + N2s1

#how big is the probability of observing >=max(N1s2,N2s1) times that one classifier is better than the other,
#in N samples, while you expected to observe that one classifier is better (N/2) times?
from scipy.stats import binom
pval = binom.cdf(min(N1s2,N2s1), N, 0.5) + (1 - binom.cdf(max(N1s2,N2s1)-1, N, 0.5)) # 0.0029
