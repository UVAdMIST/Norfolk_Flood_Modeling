import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature

df_csv = pd.read_csv('F:\RandomForest/for_RF.csv', delimiter=',')
df = df_csv.drop(['Long','Lat','event_date'],axis=1)
date = 'Aug18_21_18'
print(df_csv[df_csv['event_date']== date])


test_df = df_csv[df_csv['event_date']==date]
print(len(test_df))
train_df = df_csv[df_csv['event_date']!=date]
print(len(train_df))

test_indices = df_csv.index[df_csv['event_date']==date].tolist()
print(len(test_indices))

Test_Y = test_df['f_nf']
Test_X = test_df.drop(['Long','Lat','event_date','f_nf'], axis = 1)
print(Test_X)

f_train = train_df[train_df['f_nf']==1] #flood samples in training
nf_train = train_df[train_df['f_nf']==0] #non-flood samples in training
n = 3
NoSS = len(nf_train)/(n*len(f_train))
print(NoSS)
if round(NoSS)%2 == 0:
    NoS = round(NoSS)-1
else:
    NoS=round(NoSS)
# NoS = int(NoSS)
print(NoS)

prediction = []
for itr in range(0,11):
    train_df_copy = train_df.drop(['Long', 'Lat', 'event_date'], axis=1)
    for i in range(0, int(NoS)):
        if i<(NoS-1):
            print('train dataframe elements: ' + str(i))
            Tr_F = train_df_copy[train_df_copy['f_nf'] == 1]  # flood samples in training
            nf_train_df = train_df_copy[train_df_copy['f_nf'] == 0]  # non-flood samples in training
            nf_train_indices = train_df_copy.index[train_df_copy['f_nf'] == 0].tolist()  # non-flood sample indeces
            print('nf_train_indices' + str(len(nf_train_indices)))

            print(len(train_df_copy))

            # pick random indices of nf samples so that number of nf samples equals number of f samples
            nf_rand_ind = np.random.choice(nf_train_indices, size=n*len(Tr_F), replace=False).tolist()
            # print len(nf_train_ind)
            Tr_nF = df.iloc[nf_rand_ind]  # get training nf dataframe for randomly selected indeces
            #     print(Tr_nF)
            Train_df = pd.concat([Tr_F, Tr_nF])  # trainning dataset containg both f and nf samples

            print(len(Train_df['f_nf']))
            Train_X = Train_df.drop(['f_nf'], axis=1)  # training features
            Train_Y = Train_df['f_nf']  # training labels

            train_df_copy = train_df_copy.drop(labels=nf_rand_ind)  # remove indeces of nf samples used in this iteration

            rf_clf = RandomForestClassifier(n_estimators=100, random_state=7)
            rf_fit = rf_clf.fit(Train_X, Train_Y)
            rf_predict = rf_fit.predict(Test_X)

        elif i == (NoS-1):
            print('Last train dataframe elements ' + str(i))
            Tr_F = train_df_copy[train_df_copy['f_nf'] == 1]
            Tr_nF = train_df_copy[train_df_copy['f_nf'] == 0]

    #             print (Tr_nF)
            Train_df = pd.concat([Tr_F, Tr_nF])
            print(len(Train_df['f_nf']))
            Train_X = Train_df.drop(['f_nf'], axis=1)
            Train_Y = Train_df['f_nf']
            print('Last training set: '+str(len(Train_df)))

            rf_clf = RandomForestClassifier(n_estimators=100, random_state=7, class_weight='balanced')
            rf_fit = rf_clf.fit(Train_X, Train_Y)
            rf_predict = rf_fit.predict(Test_X)
            print('done')
        prediction.append(rf_predict.tolist())

import statistics
majority =[]
for lng in range(0, len(prediction[0])):
    lst2 = [item[lng] for item in prediction]
#     print(len(lst2))
    m = statistics.mode(lst2)
#     print(m)
    majority.append(m)
print(len(majority))
#Precision, Recall, False Positive and False Negative
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

g = np.savetxt('F:\RandomForest\Predict_location/%s_pred.csv'%(date),np.array(majority), delimiter=',', header='Prediction', comments='')
predict_df = pd.read_csv('F:\RandomForest\Predict_location/Aug18_21_18_pred.csv')
print(predict_df['Prediction'])
print(len(predict_df[predict_df['Prediction']==1]))

obs = df_csv.iloc[test_indices]
print(len(obs))
# print(len(pred))
obs.reset_index(drop=True, inplace=True)
predict_df.reset_index(drop=True, inplace=True)
final_df = pd.concat([obs,predict_df],axis=1)
final_df.to_csv('F:\RandomForest\Predict_location/%s_%s.csv'%(date,n), header=True)