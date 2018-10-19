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
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature
from sklearn.metrics import average_precision_score, auc
from sklearn.metrics import roc_curve, auc

df_csv = pd.read_csv('D:\Data TimeSeries (Hourly)\Classification/for_RF_model.csv', delimiter=',')
df = df_csv.drop(['OID_','Long','Lat','event_date'], axis=1)
print (len(df['f_nf']))
feat = df.drop(['f_nf'], axis = 1)
f_all = df[df['f_nf']==1]
f_indices = df.index[df['f_nf']==1].tolist()
print (len(f_indices))
print (len(f_all))
Features = list(feat.columns.values)
print(Features)

nf_all = df[df['f_nf']==0]
nf_indices = df.index[df['f_nf']==0].tolist()
print (len(nf_indices))

f_test_prop = .2
f_test = f_all.sample(frac = f_test_prop)
# print f_test
f_test_n = len(f_test['f_nf'])
print (f_test_n)
nf_test_n = int(round(.7*f_test_n/.3))
nf_test = f_all.sample(n = nf_test_n)
print (nf_test_n)
f_test_ind = np.random.choice(f_indices, f_test_n, replace = False ).tolist()
nf_test_ind = np.random.choice(nf_indices, nf_test_n, replace = False ).tolist()

print (len(f_test_ind), len(nf_test_ind))

test_ind = f_test_ind + nf_test_ind
print(len(test_ind))
test_df = df.ix[test_ind]
print(len(test_df[test_df['f_nf'] == 0]))

train_df = df.drop(labels=test_ind)
from statistics import mode

f_train = train_df[train_df['f_nf'] == 1]
nf_train = train_df[train_df['f_nf'] == 0]
NoS = len(nf_train['f_nf']) / len(f_train['f_nf'])
print(NoS)

Train = []
Test = []
prediction = []
feature_imp = []
probability = []
majority = []
for i in range(0, int(NoS)):
    print('train dataframe elements' + str(i))
    Tr_F = train_df[train_df['f_nf'] == 1]
    nf_train_df = train_df[train_df['f_nf'] == 0]
    nf_train_indices = train_df.index[train_df['f_nf'] == 0].tolist()
    print('nf_train_indices' + str(len(nf_train_indices)))

    print(len(train_df['f_nf']))
    # Tr_F = f_train
    nf_train_ind = np.random.choice(nf_train_indices, size=len(f_train['f_nf']), replace=False).tolist()
    # print len(nf_train_ind)
    Tr_nF = df.ix[nf_train_ind]
    # print Tr_nF
    Train_df = pd.concat([Tr_F, Tr_nF])
    print(len(Train_df['f_nf']))
    Train_X = Train_df.drop(['f_nf'], axis=1)
    Train_Y = Train_df['f_nf']
    train_df = train_df.drop(labels=nf_train_ind)

    Test_X = test_df.drop(['f_nf'], axis=1)
    Test_Y = test_df['f_nf']
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=7, class_weight='balanced')
    rf_fit = rf_clf.fit(Train_X, Train_Y)
    rf_predict = rf_fit.predict(Test_X)
    fuzzy_predict = rf_fit.predict_proba(Test_X)
    importance = []
    for b, imp in zip(Features, rf_fit.feature_importances_):
        #         print('{b} importance: {imp}'.format(b=b, imp=imp.round(3)))
        # importance.append('{b} importance: {imp} \n'.format(b=b, imp=imp.round(3)))
        importance.append(imp.round(3))

    for k in Test_Y:
        Test.append(k)
    prediction.append(rf_predict.tolist())

    for l in fuzzy_predict:
        probability.append(l)
    feature_imp.append(importance)
np.savetxt('F:\WAZE/New folder//Importance.csv', feature_imp, delimiter=',', header=str(Features), comments='')
majority =[]
for lng in range(0,len(prediction[0])):
    lst2 = [item[lng] for item in prediction]
    m = mode(lst2)
    print(m)
    majority.append(m)

print(len(Test))
print(len(prediction))
print(confusion_matrix(Test_Y.tolist(), majority))
TN = confusion_matrix(Test_Y.tolist(), majority)[0][0]
FP = confusion_matrix(Test_Y.tolist(), majority)[0][1]
FN = confusion_matrix(Test_Y.tolist(), majority)[1][0]
TP = confusion_matrix(Test_Y.tolist(), majority)[1][1]
print(TP, TN, FP, FN)

FP_rate = FP * 100 / (TN + FP)
FN_rate = FN * 100 / (TP + FN)
print("FP rate " + str(FP_rate))
print("FN rate " + str(FN_rate))

print('non-flood precision: ' + str(precision_score(Test_Y.tolist(), majority, pos_label=0, average='binary')))
print('flood precision: ' + str(precision_score(Test_Y.tolist(), majority, pos_label=1, average='binary')))

print('non-flood recall: ' + str(recall_score(Test_Y.tolist(), majority, pos_label=0, average='binary')))
print('flood recall: ' + str(recall_score(Test_Y.tolist(), majority, pos_label=1, average='binary')))
Probab_f = []
for p in probability:
    #     print (p)
    Probab_f.append(p[1])
#precision-recall curve
average_precision = average_precision_score(Test, Probab_f)
precision, recall, threshold = precision_recall_curve(Test, Probab_f,  pos_label= 1)
auc_pr = auc(recall, precision)
# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve: Area={0:0.2f}'.format(
          auc_pr))
plt.savefig('F:/PR curve.jpg', dpi = 300)
print(auc_pr)
print(average_precision)

#ROC curve
fpr, tpr, threshold = roc_curve(Test, Probab_f, pos_label=1)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 1
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")
plt.savefig('F:/ROC curve.jpg', dpi = 300)
plt.show()