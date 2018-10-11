# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 16:42:22 2018

@author: Ifrana
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

df_csv = pd.read_csv('D:\Data TimeSeries (Hourly)\Classification/for_RF_model.csv', delimiter=',')
df = df_csv.drop(['OID_','Long','Lat','event_date'], axis=1)
Label = df["f_nf"]
print (len(df['f_nf']))
Variables = df.drop(['f_nf'], axis=1)

f_all = df[df['f_nf']==1]
f_indices = df.index[df['f_nf']==1].tolist()
print (len(f_indices))
print (len(f_all))

nf_all = df[df['f_nf']==0]
nf_indices = df.index[df['f_nf']==0].tolist()
print (len(nf_indices))

f_test_prop = .2
f_test = f_all.sample(frac = f_test_prop)
# print f_test
f_test_n = len(f_test['f_nf'])
print (f_test_n)
nf_test_n = int(round(.6*f_test_n/.4))
nf_test = f_all.sample(n = nf_test_n)
print (nf_test_n)
f_test_ind = np.random.choice(f_indices, f_test_n, replace = False ).tolist()
nf_test_ind = np.random.choice(nf_indices, nf_test_n, replace = False ).tolist()

print (len(f_test_ind), len(nf_test_ind))
#print np.shape(nf_test_ind)

# print df.ix[f_indices]
test_ind = f_test_ind+nf_test_ind
print (len(test_ind))
test_df = df.ix[test_ind]
print (len(test_df[test_df['f_nf']==0]))

train_df = df.drop(labels=test_ind)


f_train = train_df[train_df['f_nf']==1]
nf_train = train_df[train_df['f_nf']==0]
# nf_train_indices = train_df.index[train_df['f_nf']==0].tolist()
# print len(nf_train_indices)
# print f_train
# print nf_train

NoS = len(nf_train['f_nf'])/len(f_train['f_nf'])
print (NoS)

Train = []
Test = []
prediction = []
feature_imp = []
probab = []

for i in range(0, 22):
    Tr_F = train_df[train_df['f_nf'] == 1]
    nf_train_df = train_df[train_df['f_nf'] == 0]
    nf_train_indices = train_df.index[train_df['f_nf'] == 0].tolist()
    print (len(nf_train_indices))

    print ('train dataframe elements' + str(i))
    print (len(train_df['f_nf']))
    # Tr_F = f_train
    nf_train_ind = np.random.choice(nf_train_indices, size = len(f_train['f_nf']), replace=False).tolist()
    # print len(nf_train_ind)
    Tr_nF = df.ix[nf_train_ind]
    # print Tr_nF
    Train_df = pd.concat([Tr_F,Tr_nF])
    print (len(Train_df['f_nf']))
    Train_X = train_df.drop(['f_nf'],axis = 1)
    Train_Y = train_df['f_nf']
    train_df = train_df.drop(labels = nf_train_ind)
    
    Test_X = test_df.drop(['f_nf'],axis = 1)
    Test_Y = test_df['f_nf']
    rf_clf =  RandomForestClassifier(n_estimators= 100, random_state = 7, class_weight = 'balanced')
    rf_fit = rf_clf.fit(Train_X, Train_Y)
    rf_predict = rf_fit.predict(Test_X)git
    for i in Test_Y:
        Test.append(i) 
    for j in rf_predict:
        prediction.append(j)
print (len(Test))
print (len(prediction))
print(confusion_matrix(Test, prediction))
TN = confusion_matrix(Test, prediction)[0][0]
FP = confusion_matrix(Test, prediction)[0][1]
FN = confusion_matrix(Test, prediction)[1][0]
TP = confusion_matrix(Test, prediction)[1][1]
print(TP, TN, FP, FN)

FP_rate = FP*100/(TN+FP)
FN_rate = FN*100/(TP+FN)
print ("FP rate "+str(FP_rate))
print ("FN rate "+str(FN_rate))

print(precision_score(Test, prediction, average='binary'))

print(recall_score(Test, prediction, average='binary'))
