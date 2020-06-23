# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 23:47:34 2020

@author: ravit
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('glassdoor_jobs_data_cleaned.csv')

# chose relevant columns
df.columns

df_model = df[['avg_salary','Rating', 'Size', 'Type of ownership', 'Industry', 'Sector', 'Revenue', 'comp_count', 'hourly',
             'employer_provided', 'job_state', 'age_of_company', 'python_yn', 'spark_yn', 'aws_yn', 'excel_yn', 'job_simp',
             'seniority', 'job_desc_len']]

# get dummy data
df_dum = pd.get_dummies(df_model)

# train test split
from sklearn.model_selection import train_test_split

X = df_dum.drop('avg_salary', axis=1)
y = df_dum.avg_salary.values  #df_dum.avg_salary gives series, while the current one generates an array
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

# multiple linear regression
import statsmodels.api as sm

X_sm = X = sm.add_constant(X)
model = sm.OLS(y,X_sm)
model.fit().summary()

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score

lm = LinearRegression()
lm.fit(X_train, y_train)

np.mean(cross_val_score(lm, X_train, y_train, scoring='neg_mean_absolute_error', cv=3))

# lasso regression
lm_l = Lasso()
lm_l.fit(X_train, y_train)

np.mean(cross_val_score(lm_l, X_train, y_train, scoring='neg_mean_absolute_error', cv=3))

alpha = []
err = []

for i in range(1,100):
    alpha.append(i/100)
    lml = Lasso(alpha = (i/100))
    err.append(np.mean(cross_val_score(lml, X_train, y_train, scoring='neg_mean_absolute_error', cv=3)))
    
plt.plot(alpha, err) 

error = tuple(zip(alpha, err))
df_err = pd.DataFrame(error, columns = ['alpha', 'err'])
df_err[df_err.err == max(df_err.err)]

lm_2 = Lasso(alpha = 0.03) #fit with alpha obtained from the above value
lm_2.fit(X_train, y_train)

np.mean(cross_val_score(lm_2, X_train, y_train, scoring='neg_mean_absolute_error', cv=3))

# random forest
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rf.fit(X_train, y_train)
np.mean(cross_val_score(rf, X_train, y_train, scoring='neg_mean_absolute_error', cv=3))

# tune models GridsearchCV
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators': range(10, 300, 10), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')}

gs = GridSearchCV(rf, parameters,scoring='neg_mean_absolute_error', cv=3)
gs.fit(X_train, y_train)

gs.best_score_
gs.best_estimator_

# test ensembles
tpred_lm = lm.predict(X_test)
tpred_lm_2 = lm_2.predict(X_test)
tpred_rf = gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, tpred_lm)
mean_absolute_error(y_test, tpred_lm_2)
mean_absolute_error(y_test, tpred_rf)

mean_absolute_error(y_test, (tpred_lm+tpred_rf)/2)

import pickle
pickl = {'model':gs.best_estimator_}
pickle.dump(pickl, open('model_file' + ".p", "wb"))

file_name = "model_file.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']
    
model.predict(X_test.iloc[1,:].values.reshape(1,-1))
list(X_test.iloc[1,:])
