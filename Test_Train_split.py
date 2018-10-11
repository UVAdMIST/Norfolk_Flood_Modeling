# -*- coding: utf-8 -*-
"""
This program executes a train/test split, random forest classification, and accuracy assessment.
   Authors:    G. O'Neil and F.Zahura
   Changelog: 20180323: Initial version
"""
from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import RandomForestRegressor
import numpy as np
#from sklearn import metrics
#from sklearn import model_selection
#from sklearn.metrics import *
#import os
#import matplotlib.pyplot as plt
#import sys
import pandas as pd
#from io import StringIO
#import ntpath
#import time
#import scipy
#from scipy import stats
#import matplotlib.pyplot as plt

    
def create_tt_labels(verif_array, f_train_prop, nf_train_prop):
    
    """
    input_feat = input features that will be used to differentiate between true land classes, \ 
           should be array-like where dimensions correspond to feature categories
      
    verif_data = verification data that is used to create training and testing data
    
    w_train_prop, nw_train_prop = float between 0.0 and 1.0 that represents the proportion of the wetland
        and nonwetland samples that will be used for training, the complement of these variables will \
        be the testing size
    
    """    
    print ("Creating training and testing data..." + '\n')
    
    """Training and testing LABEL creation (0s and 1s)"""
    #wetlands = 0 and nonwetlands = 1 in verificaiton dataset
    f_all = np.ma.masked_values(verif_array, 0.)
    nf_all = np.ma.masked_where(verif_array>0, verif_array)
    print ("f_all, nf_all")
    print (f_all, nf_all)
    print (np.unique(f_all), np.unique(nf_all))
    


#    #get all wetland indices (ie, cannot choose from masked/NaN elements)
    f_indices = np.where(f_all >= 1)
    nf_indices = np.where(nf_all == 0)
    print("f_indices, nf_indices")
#    print (f_indices, len(f_indices[0]))
#    print (nf_indices, len(nf_indices[0])) 
#    #get total number of wetland and nonwetland features and calculate number of samples needed
    f_total = float(len(f_indices[0]))
    f_train_n = float(f_train_prop * f_total)
    
    nf_total = float(len(nf_indices[0]))
    nf_train_n = float(nf_train_prop * nf_total)
    print("f_total, nf_total")
    print (f_total, nf_total)
    print(f_train_n, nf_train_n)
    
    
#    print "Total number of verification wetland samples: %d" %(int(w_total)) + '\n'
#    print "Total number of verification nonwetland samples %d" %(int(nw_total)) + '\n'
#    
    #choose random indices from wetland and nonwetland arrays to use for training
    #NOTE: np.where returns a list of data, first index is the array we want
    #np.random.seed(7)
    f_rand_indices = np.random.choice(f_indices[0], size = int(f_train_n), replace = False)
    nf_rand_indices = np.random.choice(nf_indices[0], size = int(nf_train_n), replace = False)
    
    #create empty boolean arrays where all elements are False
    f_temp_bool = np.zeros(f_all.shape, bool)
    nf_temp_bool = np.zeros(nf_all.shape, bool)

    #make elements True where the random indices exist
    f_temp_bool[f_rand_indices] = True
    nf_temp_bool[nf_rand_indices] = True
    
#    print( f_temp_bool, nf_temp_bool)

    #training samples are created from true values of the random indices    
    f_train = np.ma.masked_where(f_temp_bool == False, f_all)   
    nf_train = np.ma.masked_where(nf_temp_bool == False, nf_all)
    
#    print(f_train, nf_train)
    
#    sys.exit()
    #testing samples are created from the false values of the random indices (complement)
    f_test = np.ma.masked_where(f_temp_bool == True, f_all)   
    nf_test = np.ma.masked_where(nf_temp_bool == True, nf_all)
    
#    print(f_test,nf_test)
    
#    w_train_flt, nw_train_flt = w_train.astype(float), nw_train.astype(float)
#    w_test_flt, nw_test_flt = w_test.astype(float), nw_test.astype(float)
#    
    #convert masked elements to NaN
    f_train_nans = np.ma.filled(f_train, -9999)
    nf_train_nans = np.ma.filled(nf_train, -9999)
    f_test_nans = np.ma.filled(f_test, -9999)
    nf_test_nans = np.ma.filled(nf_test, -9999)
    
#    print (f_test_nans, nf_test_nans)
    #combine training wetlands and nonwetladns into a single array, reshape to write as geotiff
    train_labels = np.where(nf_train_nans > -9999, nf_train_nans, f_train_nans)
    
    #combine testing wetlands and nonwetlands into a single array, reshape to write as geotiff    
    test_labels = np.where(nf_test_nans> -9999, nf_test_nans, f_test_nans)

#    #stats
#    true_ratio = float(w_total / nw_total)
#    train_ratio = float(w_train_n / nw_train_n)
#    
#    print "True wetlands to nonwetlands ratio: %.3f" %(true_ratio) +'\n'
#    print "Training wetlands to nonwetlands ratio: %.3f" %(train_ratio) +'\n'
#    
#    train_labels_2d = np.reshape(train_labels, (verif_data.shape[0], verif_data.shape[1]))
#    test_labels_2d = np.reshape(test_labels, (verif_data.shape[0], verif_data.shape[1]))
#
#    train_labels_tif = array_to_geotif(train_labels_2d, verif_meta, out_dir, "train_w%.3f_nw%.3f.tif" %(w_train_prop, nw_train_prop))
#    test_labels_tif = array_to_geotif(test_labels_2d, verif_meta, out_dir,"test_w%.3f_nw%.3f.tif" %(float(1-w_train_prop), float(1-nw_train_prop)))
#
    return train_labels, test_labels
#
def create_tt_feats(feat_arr, train_labels, test_labels):
    
    """Training and testing FEATURES creation (input variables that are labeled either 0 or 1)"""
    #create arrays of input variables within training and testing limits    
    train_features = np.copy(feat_arr)
    test_features = np.copy(feat_arr)
    print("train feature")
    train_features[ train_labels == -9999 , :  ] = -9999
    test_features[ test_labels == -9999, :  ] = -9999    
    print(np.shape(train_features))
    
    
    return train_features, test_features

def classify(train_features, train_labels, test_features, feat_name, n_trees = 100, tree_depth =None):

    #Initialize RF model

#    print "Initializing Random Forest model with:" 
#    print "%d trees" %(n_trees)
#    print "%s max tree depth" %str(tree_depth)
#    print "class weights: W = %s | NW = %s" %(str(wcw), str(nwcw)) + '\n'
#    
#    #train RF model
    rf_clf =  RandomForestClassifier(n_estimators= n_trees, max_depth = tree_depth, random_state = 7)
    
    train_X = train_features[ train_features[:,0] > -9999, :]
    print(train_features[:,0])
    print (np.shape(train_X))

    train_Y = train_labels[train_labels > -9999] 
    print (np.shape(train_Y))
    print(len(train_Y))
#    train_Y = train_labels_2d[~np.isnan(train_labels_2d)] 

#    train_Y1 = train_Y.astype(int)
#    train_Y2 = train_Y1.astype(str)

    #train_X is an array of nonNaN values from train_features, 3D array --> 2D array ([X*Y, Z])
    #train_Y is an array of nonNaN values from train_labels, 2D array --> 1D array (X*Y)
    #both have NO NaN VALUES, which is required for sklearn rf    

#    print "Training model..."
    rf_fit = rf_clf.fit(train_X, train_Y)    

    #save feature importance    
    
#    n_feats = len(rf_fit.feature_importances_)
    importance=[]
    #bands = np.arange(1, n_feats+1)
    
    for b, imp in zip(feat_name, rf_fit.feature_importances_):
        print('{b} importance: {imp}'.format(b=b, imp=imp.round(3)))
        #importance.append('{b} importance: {imp} \n'.format(b=b, imp=imp.round(3)))
        importance.append(imp.round(3))
    
    #execute RF classification and fuzzy classification


    #cannot delete NaN values of test_features, because we will not be able to export to geotiff with extents of original ROI
    #convert test_features to 2D and fill nan values with other # (-9999)
    

    #the array used in rf.predict() and rf.predict_proba() MUST be the same size as the original ROI, eventhough those extents include 
    #many NAN values. Here, we assign a dummy value (-9999) to NaN locations to maintain shape. There is no training data for these
    #values, so the accuracy for this class should be zero but not affect the accuracy of W/NW classes

#    print "\n" + "Executing prediction...\n"

    
    rf_predict = rf_fit.predict(test_features)
#    fuzzy_predict = rf_fit.predict_proba(test_features)

    
    #reshape outputs to prepare for geotif export
#    rf_predict = rf_predict(test_features[:, :, 0].shape)
    
#    fuzzy_predict_w = fuzzy_predict[:,0].reshape(test_features[:, :, 0].shape)	
    
    
    return rf_predict, importance

df = pd.read_csv('D:/Data TimeSeries (Hourly)/for_RFmodel_hourly -LW_Trial1.csv', delimiter=',')
Dummy = df["num_flooded"]
Dummy1 = df.drop(['num_flooded','event_date'], axis=1)
Features = list(Dummy1.columns.values)

Train = []
Test = []
prediction = []
feature_imp = []

Train_prop = 0.1
for i in range(0,100):
    train, test = create_tt_labels(Dummy,0.8,Train_prop)
    for i in test:
        Test.append(i)  
     
    train_f, test_f = create_tt_feats(Dummy1, train, test)
    rf_pred, imp = classify(train_f, train, test_f, Features)
    

    for j in rf_pred:
        prediction.append(j)

    feature_imp.append(imp)
        

#print(train)
#print(Test)
#print(train_f)
#print(rf_pred)

 
np.savetxt("D:/Data TimeSeries (Hourly)/Under_Sampling/%s_test_target_LW.csv" %(Train_prop), Test,delimiter = ',',header='num_flooded')
np.savetxt("D:/Data TimeSeries (Hourly)/Under_Sampling/%s_train_target_LW.csv" %(Train_prop), train,delimiter = ',')
np.savetxt("D:/Data TimeSeries (Hourly)/Under_Sampling/%s_train_feature_LW.csv" %(Train_prop), train_f,delimiter = ',')
np.savetxt("D:/Data TimeSeries (Hourly)/Under_Sampling/%s_test_feature_LW.csv" %(Train_prop), test_f,delimiter = ',')
np.savetxt('D:/Data TimeSeries (Hourly)/Under_Sampling/%s_prediction_test_LW.csv' %(Train_prop), prediction ,delimiter = ',',header='Prediction' )

np.savetxt('D:/Data TimeSeries (Hourly)/Under_Sampling/%s_imp_test_LW.csv' %(Train_prop), feature_imp , newline='\n', delimiter = ',', header= str(Features), comments='')
