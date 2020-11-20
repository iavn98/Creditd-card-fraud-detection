# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 20:10:15 2019

@author: Gankinck
"""
import pandas as pd  
import numpy as np   
from sklearn.metrics import roc_auc_score  
from sklearn.model_selection import train_test_split 
import lightgbm as lgb  
from sklearn import metrics
import datetime
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from imblearn.datasets import make_imbalance

 

def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat), True

def data_preprocess():
    data_path = 'c:/data/'
    df_train = pd.read_csv(data_path + 'train.csv')
    df_test = pd.read_csv(data_path + 'test.csv')
    drop_item = ['txkey']
    change_item = ['ecfg', 'flbmk', 'flg_3dsmk', 'insfg', 'ovrlt']
    item_name = ['etymd', 'contp', 'ecfg', 'flg_3dsmk', 'hcefg', 'ovrlt', 'insfg', 'flbmk', 'iterm', 'stscd']    
    ave_item_name = ['ave_etymd', 'ave_contp', 'ave_ecfg', 'ave_flg_3dsmk', 'ave_hcefg',
                     'ave_ovrlt', 'ave_insfg', 'ave_flbmk', 'ave_iterm', 'ave_stscd']#盜刷率
    one_hot_list = ['acqic', 'mchno', 'mcc','stocn', 'ecfg', 'flbmk', 'flg_3dsmk', 
                    'insfg', 'ovrlt', 'hcefg', 'etymd', 'contp', 'stscd']
    df_train['flbmk'].fillna('N', inplace=True)
    df_train['flg_3dsmk'].fillna('N', inplace=True)
    df_test['flbmk'].fillna('N', inplace=True)
    df_test['flg_3dsmk'].fillna('N', inplace=True)
    df_train['conam'] = np.log1p(df_train['conam'])   
    df_train['hour'] = df_train['loctm'].apply(lambda x: (x // 10000))
    df_train['mintue'] = df_train['loctm'].apply(lambda x: (x % 10000)//100)
    df_train['sec'] = df_train['loctm'].apply(lambda x: (x % 10000)%100)
    df_test['conam'] = np.log1p(df_test['conam'])
    df_test['hour'] = df_test['loctm'].apply(lambda x: (x // 10000))
    df_test['mintue'] = df_test['loctm'].apply(lambda x: (x % 10000)//100)
    df_test['sec'] = df_test['loctm'].apply(lambda x: (x % 10000)%100)   
    for i in drop_item:
        df_train = df_train.drop([i], axis = 1)
        df_test = df_test.drop([i], axis = 1)
         
    for i in change_item:
        df_train[i] = df_train[i].apply(lambda x:0 if x == 'N' else 1)
        df_test[i] = df_test[i].apply(lambda x:0 if x == 'N' else 1)
        
    for i, z in zip(item_name, ave_item_name): 
        ave_etymd_fraud_id = df_train[[i, 'fraud_ind']].groupby([i],
                                     as_index = False).mean().sort_values(by = 'fraud_ind',ascending = False)
        ave_etymd_fraud_id.columns = [i, z]
        df_train = pd.merge(df_train, ave_etymd_fraud_id, how = 'left', on = [i])
        df_test = pd.merge(df_test, ave_etymd_fraud_id, how = 'left', on = [i])
    
    y = df_train['fraud_ind']
    y = pd.DataFrame(y)      
    df_train = df_train.drop(['fraud_ind'], axis = 1)
    return df_train, df_test, y
    


df_train, df_test, y = data_preprocess()




def data_input():
    data_path = 'c:/data/'
    df_train = pd.read_csv(data_path + 'train_10_12.csv')
    df_test = pd.read_csv(data_path + 'test_10_12.csv')
    y = df_train['fraud_ind']
    y = pd.DataFrame(y)      
    df_train = df_train.drop(['fraud_ind'], axis = 1)
    df_test = df_test.drop(['fraud_ind'], axis = 1)
    return df_train, df_test, y

df_train, df_test, y = data_input()
def smote():
    sm = SMOTE(random_state = 42)
    df_size, y_size = sm.fit_sample(df_train, y)
    n_sample = y_size.shape[0]
    n_pos_sample = y_size[y_size == 0].shape[0]
    n_neg_sample = y_size[y_size == 1].shape[0]
    print('样本个数：{}; 正样本占{:.2%}; 负样本占{:.2%}'.format(n_sample,
                                                       n_pos_sample / n_sample,
                                                       n_neg_sample / n_sample))
    return df_size, y_size
    
df_size, y_size = smote()

def enn():   
    rus = RandomUnderSampler(random_state=0)
    x_sample, y_sample = rus.fit_sample(df_train, y)
    return  x_sample, y_sample

x_sample, y_sample = enn()

y_sample = y_sample.ravel()
def light_gbm_predict(df_size, y_size, leave = 350, depth = 20, learning = 0.047,
                      boost_round = 1784 , feature = 0.74, bagging = 0.826):  
    '''
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param leave : [250, 800]
    :param depth: [12, 40]
    :param learning:[0.5, 0.0001]
    :param boost_round = [1000, 6000]
    :param feature = [0.7, 0.95]
    :param bagging = [0.7, 0.95]
    :return:
    '''
    lgb_train = lgb.Dataset(df_size, y_size) # create dataset for lightgbm   
    lgb_eval = lgb.Dataset(df_size, y_size, reference = lgb_train) # create dataset for lightgbm  
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'AUC',
        'num_leaves': leave,
        'max_depth': depth,
        'learning_rate': learning,
        'feature_fraction': feature,
        'bagging_fraction': bagging,
        'min_data_in_leaf': 20,
        'bagging_freq': 5,
        'verbose': -1,
        
    }
    print('Start training...')  
    gbm = lgb.train(params,  
                    lgb_train,
                    num_boost_round = boost_round,  
                    valid_sets=lgb_eval,
                    feval = lgb_f1_score,
                    early_stopping_rounds = 1000)
    y_pred = gbm.predict(df_test, num_iteration = gbm.best_iteration)
    data_path = 'c:/data/'
    submission = pd.read_csv(data_path + 'submission_test.csv')
    submission.iloc[:,1] = y_pred
    return submission

submission_test = light_gbm_predict(df_train, y)
submission_prob = submission_test['fraud_ind'].apply(lambda x:1 if x > 0.1 else 0)
data_path = 'c:/data/'
submission = pd.read_csv(data_path + 'submission_test.csv')
submission.iloc[:,1] = submission_prob
submission_prob  = pd.DataFrame(submission_prob)  
submission_1 = submission_prob[submission_prob['fraud_ind'] == 1]
submission.to_csv("smote_fraud_ind_10_13_2.csv", index=False)
df_train.to_csv("train_10_13.csv", index=False)
df_test.to_csv("test_10_13.csv", index=False)


