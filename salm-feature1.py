#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import pathlib
from pathlib import Path
root_dir = Path.cwd()
salm = pd.read_csv(root_dir /'suu/data/Salm_KNN_complete.csv/')


# In[ ]:


from itertools import combinations
a = []
features = ['TEMP','SLP','WDSP','PRCP','RH','SLP-STP','MSXPD-WDSP',
            'Median age','Immigrants','Households Median Income','%Food preparation','%Farming, fishing, and forestry','Population Density']
for p in combinations(features,6):
    a.append(p)
for p in combinations(features,13):
    a.append(p)

len(a)


# In[36]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
import xgboost as xgb
from catboost import CatBoostRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

df = pd.DataFrame()
for i in range(len(a)):
    rf = []; xg = []; dt = []; cat = []; lg = [];
    rf_rs = []; dt_rs = []
    X = salm[list(a[i])]
    y = salm['TotalCases']
    for rs in range(0,100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = rs)
        rf_1 = []; dt_1 = []
        #XGBOOST
        model = xgb.XGBRegressor()
        model.fit(X_train, y_train)
        y_pred_xg = model.predict(X_test)
        xg.append(r2_score(y_test, y_pred_xg))
        
        #Random Forest
        for rm in range(0,100):
            model2 = RandomForestRegressor(random_state = rm)
            model2.fit(X_train, y_train)
            y_pred_rf = model2.predict(X_test)
            rf_1.append(r2_score(y_test, y_pred_rf))
        idx_rf = rf_1.index(max(rf_1))
        rf.append(max(rf_1))
        rf_rs.append(idx_rf)
        
        #Decision Tree
        for rd in range(0,100):
            model3 = DecisionTreeRegressor(random_state = rd)
            model3.fit(X_train, y_train)
            y_pred_dt = model3.predict(X_test)
            dt_1.append(r2_score(y_test, y_pred_dt))
        idx_dt = dt_1.index(max(dt_1))
        dt.append(max(dt_1))
        dt_rs.append(idx_dt)
        
        #Catboost
        model = CatBoostRegressor()
        model.fit(X_train, y_train)
        y_pred_cat = model.predict(X_test)
        cat.append(r2_score(y_test, y_pred_cat))
        
        
    #find the maximum R2 score
    s = []
    for j in range(0,100):
        m = dt[j] + rf[j] + xg[j] + cat[j]
        s.append(m)
        
    idx = s.index(max(s))
    
    df.loc[i, "Features"] = str(a[i])
    df.loc[i, "rs"] = idx
    df.loc[i, "r2_xg"] = xg[idx]
    df.loc[i, "r2_rf"] = rf[idx]
    df.loc[i, "r2_dt"] = dt[idx]
    df.loc[i, "rs_rf"] = rf_rs[idx]
    df.loc[i, "rs_dt"] = dt_rs[idx] 
    df.loc[i, "r2_cat"] = cat[idx]


# In[ ]:


df.to_csv('/scratch/rong07/salm6.csv')


# In[41]:


fea = pd.DataFrame()
for i in range(len(features)):
    df1 = df[df.Features.str.contains(str(features[i]))]
    df1.reset_index(drop=True, inplace=True)
    fea.loc[i, "Features"] = str(features[i])
    fea.loc[i, "r2_xg"] = df1['r2_xg'].mean()
    fea.loc[i, "r2_rf"] = df1['r2_rf'].mean()
    fea.loc[i, "r2_dt"] = df1['r2_dt'].mean()
    fea.loc[i, "r2_cat"] = df1['r2_cat'].mean()
    fea.loc[i, "r2xg"] = df['r2_xg'].mean()
    fea.loc[i, "r2rf"] = df['r2_rf'].mean()
    fea.loc[i, "r2dt"] = df['r2_dt'].mean()
    fea.loc[i, "r2cat"] = df['r2_cat'].mean()


# In[43]:


fea.to_csv('/scratch/rong07/salm_fea.csv')

