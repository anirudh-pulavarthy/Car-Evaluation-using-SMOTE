'''
created by Anirudh Pulavarthy
Adapted homework code by Dr. Casey Bennett for DSC 540
'''

import sys
import csv
import math
import numpy as np
from operator import itemgetter
import time

from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
import joblib
from sklearn.feature_selection import RFE, VarianceThreshold, SelectFromModel
from sklearn.feature_selection import SelectKBest, mutual_info_regression, mutual_info_classif, chi2
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import KBinsDiscretizer, scale
from sklearn.preprocessing import OrdinalEncoder
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

#Handle annoying warnings
import warnings, sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)


#############################################################################
#
# Global parameters
#
#####################

target_idx=6                                        #Index of Target variable
cross_val=0                                         #Control Switch for CV                                                                                                                                                      
norm_target=0                                       #Normalize target switch
norm_features=0                                     #Normalize target switch
binning=1                                           #Control Switch for Bin Target
bin_cnt=2                                           #If bin target, this sets number of classes
feat_select=0                                       #Control Switch for Feature Selection                                                                                   
fs_type=2                                           #Feature Selection type (1=Stepwise Backwards Removal, 2=Wrapper Select, 3=Univariate Selection)                        
lv_filter=0                                         #Control switch for low variance filter on features
feat_start=0                                        #Start column of features
k_cnt=5                                             #Number of 'Top k' best ranked features to select, only applies for fs_types 1 and 3
model=1                                             #Indicates which model to use for train/test split: 1=Random Forests, 2=MLP Classifier,
                                                    # 3=GradientBoosting, 4 = AdaBoost

smote=1                                             #Control switch for smote
                                                    #

#Set global model parameters
rand_st=1                                           #Set Random State variable for randomizing splits on runs


#############################################################################
#
# Load Data
#
#####################

file1= csv.reader(open('/Users/anirudh/Desktop/DSC540/FinalProject/car.data'), delimiter=',', quotechar='"')

#These are the attribute names
header= ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

#Read data
data=[]
target=[]

for row in file1:
    #Load Target
    if row[target_idx]=='':                         #If target is blank, skip row                       
        continue
    else:
        target.append(row[target_idx])              #If pre-binned class, change float to int #anirudh

    #Load row into temp array, cast columns  
    temp=[]
                 
    for j in range(feat_start,len(header) - 1):     # -1 excludes target from the data
        if row[j]=='':
            temp.append(float())
        else:
            temp.append(row[j])

    #Load temp into Data array
    data.append(temp)


# define ordinal encoding
encoder = OrdinalEncoder()
# transform data
data_old = data
data = encoder.fit_transform(data)
  
#Test Print
print(header)
print(len(target),len(data))
print('\n')

data_np=np.asarray(data)
target_np=np.asarray(target)


#############################################################################
#
# Preprocess data
#
##########################################

if norm_target==1:
    #Target normalization for continuous values
    target_np=scale(target_np)

if norm_features==1:
    #Feature normalization for continuous values
    data_np=scale(data_np)

'''if binning==1:
    #Discretize Target variable with KBinsDiscretizer
    enc = KBinsDiscretizer(n_bins=[bin_cnt], encode='ordinal', strategy='quantile')                         #Strategy here is important, quantile creating equal bins, but kmeans prob being more valid "clusters"
    target_np_bin = enc.fit_transform(target_np.reshape(-1,1))

    #Get Bin min/max
    temp=[[] for x in range(bin_cnt+1)]
    for i in range(len(target_np)):
        for j in range(bin_cnt):
            if target_np_bin[i]==j:
                temp[j].append(target_np[i])

    for j in range(bin_cnt):
        print('Bin', j, ':', min(temp[j]), max(temp[j]), len(temp[j]))
    print('\n')

    #Convert Target array back to correct shape
    target_np=np.ravel(target_np_bin)'''


#############################################################################
#
# Feature Selection
#
##########################################

#Low Variance Filter
if lv_filter==1:
    print('--LOW VARIANCE FILTER ON--', '\n')
    
    #LV Threshold
    sel = VarianceThreshold(threshold=0.5)                                      #Removes any feature with less than 20% variance
    fit_mod=sel.fit(data_np)
    fitted=sel.transform(data_np)
    sel_idx=fit_mod.get_support()

    #Get lists of selected and non-selected features (names and indexes)
    temp=[]
    temp_idx=[]
    temp_del=[]
    for i in range(len(data_np[0])):
        if sel_idx[i]==1:                                                           #Selected Features get added to temp header
            temp.append(header[i+feat_start])
            temp_idx.append(i)
        else:                                                                       #Indexes of non-selected features get added to delete array
            temp_del.append(i)

    print('Selected', temp)
    print('Features (total, selected):', len(data_np[0]), len(temp))
    print('\n')

    #Filter selected columns from original dataset
    header = header[0:feat_start]
    for field in temp:
        header.append(field)
    data_np = np.delete(data_np, temp_del, axis=1)                                 #Deletes non-selected features by index



#Feature Selection
if feat_select==1:
    '''Three steps:
       1) Run Feature Selection
       2) Get lists of selected and non-selected features
       3) Filter columns from original dataset
       '''
    
    print('--FEATURE SELECTION ON--', '\n')
    
    ##1) Run Feature Selection #######
    if fs_type==1:
        #Stepwise Recursive Backwards Feature removal
        if binning==1:
            clf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=3, criterion='entropy', random_state=rand_st)
            sel = RFE(clf, n_features_to_select=k_cnt, step=.1)
            print('Stepwise Recursive Backwards - Random Forest: ')
        if binning==0:
            rgr = RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_split=3, criterion='mse', random_state=rand_st)
            sel = RFE(rgr, n_features_to_select=k_cnt, step=.1)
            print('Stepwise Recursive Backwards - Random Forest: ')
            
        fit_mod=sel.fit(data_np, target_np)
        print(sel.ranking_)
        sel_idx=fit_mod.get_support()      

    if fs_type==2:
        #Wrapper Select via model

        clf1 = GradientBoostingClassifier(loss = 'log_loss', n_estimators = 100, learning_rate = 0.1, max_depth = 3, min_samples_split = 3, random_state = rand_st) #deviance is now log_loss
        clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=3, criterion='entropy', random_state=rand_st)
        sel = SelectFromModel(clf, prefit=False, threshold='mean', max_features=None)                                                           #to select only based on max_features, set to integer value and set threshold=-np.inf
        print ('Wrapper Select: ')

        fit_mod=sel.fit(data_np, target_np)    
        sel_idx=fit_mod.get_support()

    if fs_type==3:
        if binning==1:                                                              ######Only work if the Target is binned###########
            #Univariate Feature Selection - Chi-squared
            sel=SelectKBest(chi2, k=k_cnt)
            fit_mod=sel.fit(data_np, target_np)                                         #will throw error if any negative values in features, so turn off feature normalization, or switch to mutual_info_classif
            print ('Univariate Feature Selection - Chi2: ')
            sel_idx=fit_mod.get_support()

        if binning==0:                                                              ######Only work if the Target is continuous###########
            #Univariate Feature Selection - Mutual Info Regression
            sel=SelectKBest(mutual_info_regression, k=k_cnt)
            fit_mod=sel.fit(data_np, target_np)
            print ('Univariate Feature Selection - Mutual Info: ')
            sel_idx=fit_mod.get_support()

        #Print ranked variables out sorted
        temp=[]
        scores=fit_mod.scores_
        for i in range(feat_start, len(header)):            
            temp.append([header[i], float(scores[i-feat_start])])

        print('Ranked Features')
        temp_sort=sorted(temp, key=itemgetter(1), reverse=True)
        for i in range(len(temp_sort)):
            print(i, temp_sort[i][0], ':', temp_sort[i][1])
        print('\n')

    ##2) Get lists of selected and non-selected features (names and indexes) #######
    temp=[]
    temp_idx=[]
    temp_del=[]
    for i in range(len(data_np[0])):
        if sel_idx[i]==1:                                                           #Selected Features get added to temp header
            temp.append(header[i+feat_start])
            temp_idx.append(i)
        else:                                                                       #Indexes of non-selected features get added to delete array
            temp_del.append(i)
    print('Selected', temp)
    print('Features (total/selected):', len(data_np[0]), len(temp))
    print('\n')
            
                
    ##3) Filter selected columns from original dataset #########
    header = header[0:feat_start]
    for field in temp:
        header.append(field)
    data_np = np.delete(data_np, temp_del, axis=1)                                 #Deletes non-selected features by index)
    
    

#############################################################################
#
# Train SciKit Models
#
##########################################

print('--ML Model Output--', '\n')

#Test/Train split
data_train, data_test, target_train, target_test = train_test_split(data_np, target_np, test_size=0.35)

####Classifiers####
if model==1 and cross_val==0:
    #SciKit
    scorers = {'Accuracy': 'accuracy', 'ovo': 'roc_auc_ovo', 'ovr': 'roc_auc_ovr'}

    start_ts=time.time()

    clf0 = DecisionTreeClassifier(splitter = 'random', criterion = 'entropy', random_state=rand_st)
    clf1 = RandomForestClassifier(n_estimators = 20, max_depth = None, min_samples_split = 3, criterion = 'entropy', random_state = rand_st)
    clf = MLPClassifier(activation = 'logistic', solver = 'lbfgs', max_iter = 100, hidden_layer_sizes = (50,50), random_state = rand_st)
    clf3 = GradientBoostingClassifier(loss = 'log_loss', n_estimators = 100, learning_rate = 0.1, max_depth = 3, min_samples_split = 3, random_state = rand_st) #deviance is now log_loss

    clf.fit(data_train, target_train)
    scores= cross_validate(clf, data_np, target_np, scoring=scorers, cv=5)

    #scores_ACC = clf.score(data_test, target_test)                                                                                                                          
    #print('Random Forest Acc:', scores_ACC)
    #pred_test = clf.predict_proba(data_test)
    #predicted_labels = clf.predict(data_test)
                                                                                 
    #scores_AUC = metrics.roc_auc_score(target_test, pred_test, multi_class='ovo')
    #scores_AUC2 = metrics.roc_auc_score(target_test, pred_test, multi_class='ovr')

    scores_Accuracy = scores['test_Accuracy']
    scores_AUC = scores['test_ovo']
    scores_AUC2 = scores['test_ovr']
    
    #print('Random Forest OVO:', scores_AUC)      
    #print('Random Forest OVR:', scores_AUC2)    

    print("NN Accuracy: %0.2f (+/- %0.2f)" % (scores_Accuracy.mean(), scores_Accuracy.std() * 2))       
    print("NN OVO: %0.2f (+/- %0.2f)" % (scores_AUC.mean(), scores_AUC.std() * 2))                                                                                                    
    print("NN OVR: %0.2f (+/- %0.2f)" % (scores_AUC2.mean(), scores_AUC2.std() * 2))    

    #cm = confusion_matrix(target_test, predicted_labels) 
    #print('\n Confusion matrix\n', cm)

    #print('\n Classification report\n')
    #print(classification_report(target_test, predicted_labels))
    print("Gradient Runtime:", time.time()-start_ts)

if model == 2:
    start_ts=time.time()
    clf= MLPClassifier(activation = 'logistic', solver = 'adam', alpha = 0.0001, max_iter = 100, hidden_layer_sizes = (200, 200), random_state = rand_st)
    clf.fit(data_train, target_train)

    scores_ACC = clf.score(data_test, target_test)                                                                                                                          
    print('MLP Classifier Acc:', scores_ACC)
    pred_test = clf.predict_proba(data_test)
    predicted_labels = clf.predict(data_test)
                                                                                 
    scores_AUC = metrics.roc_auc_score(target_test, pred_test, multi_class='ovo')
    scores_AUC2 = metrics.roc_auc_score(target_test, pred_test, multi_class='ovr')
    
    print('MLP Classifier OVO:', scores_AUC)      
    print('MLP Classifier OVR:', scores_AUC2)    


    cm = confusion_matrix(target_test, predicted_labels) 
    print('\n Confusion matrix\n', cm)

    print('\n Classification report\n')
    print(classification_report(target_test, predicted_labels))
    print("MLP Runtime:", time.time()-start_ts)      

if model == 3:
    start_ts=time.time()
    clf= GradientBoostingClassifier(loss = 'log_loss', n_estimators = 100, learning_rate = 0.1, max_depth = 3, min_samples_split = 3, random_state = rand_st) #deviance is now log_loss
    clf.fit(data_train, target_train)

    scores_ACC = clf.score(data_test, target_test)                                                                                                                          
    print('Gradient Boosting Classifier Acc:', scores_ACC)
    pred_test = clf.predict_proba(data_test)
    predicted_labels = clf.predict(data_test)
                                                                                 
    scores_AUC = metrics.roc_auc_score(target_test, pred_test, multi_class='ovo')
    scores_AUC2 = metrics.roc_auc_score(target_test, pred_test, multi_class='ovr')
    
    print('Gradient Boosting Classifier OVO:', scores_AUC)      
    print('Gradient Boosting Classifier OVR:', scores_AUC2)    


    cm = confusion_matrix(target_test, predicted_labels) 
    print('\n Confusion matrix\n', cm)

    print('\n Classification report\n')
    print(classification_report(target_test, predicted_labels))
    print("Gradient Boosting Runtime:", time.time()-start_ts)                          

if model == 4:
    start_ts=time.time()
    #clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=3, min_samples_leaf=1, max_features=None, random_state=rand_st)
    clf = DecisionTreeClassifier(random_state=rand_st)
    clf.fit(data_train, target_train)

    scores_ACC = clf.score(data_test, target_test)                                                                                                                          
    print('Decision Tree Classifier Acc:', scores_ACC)
    pred_test = clf.predict_proba(data_test)
    predicted_labels = clf.predict(data_test)
                                                                                 
    scores_AUC = metrics.roc_auc_score(target_test, pred_test, multi_class='ovo')
    scores_AUC2 = metrics.roc_auc_score(target_test, pred_test, multi_class='ovr')
    
    print('Decision Tree Classifier OVO:', scores_AUC)      
    print('Decision Tree Classifier OVR:', scores_AUC2)    


    cm = confusion_matrix(target_test, predicted_labels) 
    print('\n Confusion matrix\n', cm)

    print('\n Classification report\n')
    print(classification_report(target_test, predicted_labels))
    print("Decision Tree Runtime:", time.time()-start_ts)                                                                                                           
 
####Cross-Val Classifiers####
if binning==1 and cross_val==1:
    #Setup Crossval classifier scorers
    #scorers = {'Accuracy': 'accuracy', 'roc_auc': 'roc_auc'}                                                                                                                
    scorers = {'ovo': 'roc_auc_ovo', 'ovr': 'roc_auc_ovr'}

    #SciKit Gradient Boosting - Cross Val
    start_ts=time.time()
    clf= GradientBoostingClassifier(loss = 'log_loss', n_estimators = 100, learning_rate = 0.1, max_depth = 3, min_samples_split = 3, random_state = rand_st) #deviance is now log_loss
    scores= cross_validate(clf, data_np, target_np, scoring=scorers, cv=5)

    #scores_Acc = scores['test_Accuracy']                                                                                                                                    
    scores_Acc = scores['test_ovr']
    print("Gradient Boosting Acc: %0.2f (+/- %0.2f)" % (scores_Acc.mean(), scores_Acc.std() * 2))                                                                                                    
    #scores_AUC= scores['test_roc_auc']                                                                     #Only works with binary classes, not multiclass                  
    #print("Gradient Boosting AUC: %0.2f (+/- %0.2f)" % (scores_AUC.mean(), scores_AUC.std() * 2))                           
    print("CV Runtime:", time.time()-start_ts)


    #SciKit Ada Boosting - Cross Val
    start_ts=time.time()
    clf= AdaBoostClassifier(estimator = None, n_estimators = 100, learning_rate = 0.1, random_state = rand_st) #base_estimator is now estimator
    scores= cross_validate(clf, data_np, target_np, scoring=scorers, cv=5)

    #scores_Acc = scores['test_Accuracy']  
    scores_Acc = scores['test_ovo']
    print("Ada Boost Acc: %0.2f (+/- %0.2f)" % (scores_Acc.mean(), scores_Acc.std() * 2))                                                                                                    
    #scores_AUC= scores['test_roc_auc']                                                                     #Only works with binary classes, not multiclass                  
    #print("Ada Boost AUC: %0.2f (+/- %0.2f)" % (scores_AUC.mean(), scores_AUC.std() * 2))                           
    print("CV Runtime:", time.time()-start_ts)


    #SciKit Neural Network - Cross Val
    start_ts=time.time()
    clf= MLPClassifier(activation = 'logistic', solver = 'adam', alpha = 0.0001, max_iter = 100, hidden_layer_sizes = (2,), random_state = rand_st)
    scores= cross_validate(clf, data_np, target_np, scoring=scorers, cv=5)

    #scores_Acc = scores['test_Accuracy']                                                                                                                                    
    scores_Acc = scores['test_ovr']                                                                                                                                    
    print("\nMLP Classifier Acc: %0.2f (+/- %0.2f)" % (scores_Acc.mean(), scores_Acc.std() * 2))                                                                                                    
    #scores_AUC= scores['test_roc_auc']                                                                     #Only works with binary classes, not multiclass                  
    #print("MLP Classifier AUC: %0.2f (+/- %0.2f)" % (scores_AUC.mean(), scores_AUC.std() * 2))                           
    print("CV Runtime:", time.time()-start_ts) 

if smote == 1:
    start_ts = time.time()
    #model = RandomForestClassifier(n_estimators = 20, max_depth = None, min_samples_split = 3, criterion = 'entropy', random_state = rand_st)

    strategy1 = {'unacc':1210, 'good': 500, 'vgood': 500, 'acc': 500}       ## oversample the minority class(es) to set # of samples
    strategy2 = {'unacc':500, 'good': 500, 'vgood': 500, 'acc': 500}        ## undersample the majority class(es) to set # of samples

    over = SMOTE(sampling_strategy=strategy1, random_state=rand_st)
    under = RandomUnderSampler(sampling_strategy=strategy2)
    steps = [('over', over), ('under', under), ('model', model)]

    # unacc = 1210, acc = 384, good = 69, vgood = 65
    print(Counter(target_np))
    data_np, target_np = over.fit_resample(data_np, target_np)
    print(Counter(target_np))
    # 2362 after this

    print(Counter(target_np))
    data_np, target_np = under.fit_resample(data_np, target_np)
    print(Counter(target_np))
    # 1536
    
    scorers = {'Accuracy': 'accuracy', 'ovo': 'roc_auc_ovo', 'ovr': 'roc_auc_ovr'}

    clf.fit(data_np, target_np)
    scores= cross_validate(clf, data_np, target_np, scoring=scorers, cv=5)
    accuracy = scores['test_Accuracy']
    scores_AUC = scores['test_ovo']
    scores_AUC = scores['test_ovr']

    #print('Mean ROC AUC: %.3f' % scores.mean())
    print("NN with SMOTE Accuracy: %0.2f (+/- %0.2f)" % (accuracy.mean(), accuracy.std() * 2))                                                                                                    
    print("NN with SMOTE OVO: %0.2f (+/- %0.2f)" % (scores_AUC.mean(), scores_AUC.std() * 2))                                                                                                    
    print("NN with SMOTE OVR: %0.2f (+/- %0.2f)" % (scores_AUC2.mean(), scores_AUC2.std() * 2))    

    

    print("Time taken by the SMOTE sample: ", time.time() - start_ts)
