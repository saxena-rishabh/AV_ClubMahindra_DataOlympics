# -*- coding: utf-8 -*-
"""
Created on Sun May  5 23:54:00 2019

@author: Rishabh
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from dateutil.parser import parse

# Importing the dataset
data = pd.read_csv('train.csv')

data['checkin_date'] = pd.to_datetime(data.checkin_date, dayfirst=True)

data['booking_date'] = pd.to_datetime(data.booking_date,dayfirst=True)

data['checkin-booking'] = (data['checkin_date'] - data['booking_date']).dt.days


print(data.loc[data['checkin-booking'] < 0])

for i in range(len(data['checkin-booking'])):
    if data['checkin-booking'][i] < 0:
        data['checkin-booking'][i] = 0


data['checkout_date'] = pd.to_datetime(data.checkout_date,dayfirst=True)

data['checkout-checkin'] = (data['checkout_date'] - data['checkin_date']).dt.days

print(data.loc[data['checkout-checkin'] < 0])


###########################################################################
        
dataset = data[['checkin-booking', 'checkout-checkin' ,'channel_code', 'main_product_code', 'numberofadults', 'numberofchildren', 'persontravellingid', 'resort_region_code', 'resort_type_code', 'room_type_booked_code', 'roomnights', 'season_holidayed_code', 'state_code_residence', 'state_code_resort','total_pax', 'member_age_buckets','booking_type_code', 'cluster_code', 'reservationstatusid_code', 'resort_id','amount_spent_per_room_night_scaled']].copy()

dataset.fillna(0, inplace=True)

X = dataset.iloc[:, 0:-1].values

y = dataset.iloc[:, -1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_15 = LabelEncoder()
X[:, 15] = labelencoder_X_15.fit_transform(X[:, 15])
labelencoder_X_17 = LabelEncoder()
X[:, 17] = labelencoder_X_17.fit_transform(X[:, 17])
labelencoder_X_18 = LabelEncoder()
X[:, 18] = labelencoder_X_18.fit_transform(X[:, 18])
labelencoder_X_19 = LabelEncoder()
X[:, 19] = labelencoder_X_19.fit_transform(X[:, 19])


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


################## XGBOOST ####################################


import xgboost as xgb


xg_reg2 = xgb.XGBRegressor(booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.8, gamma=0.5, learning_rate=0.1,
       max_delta_step=0, max_depth=5, min_child_weight=5, missing=None,
       n_estimators=250, n_jobs=1, nthread=-1, objective='reg:linear',
       random_state=0, reg_alpha=10, reg_lambda=1, scale_pos_weight=1,
       seed=None, silent=True, subsample=0.8)

xg_reg2.fit(X_train,y_train)

y_pred = xg_reg2.predict(X_test)


from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_test, y_pred))



############### TEST DATASET #############################################

data2 = pd.read_csv('test.csv')

data2['checkin_date'] = pd.to_datetime(data2.checkin_date, dayfirst=True)

data2['booking_date'] = pd.to_datetime(data2.booking_date,dayfirst=True)

data2['checkin-booking'] = (data2['checkin_date'] - data2['booking_date']).dt.days


print(data2.loc[data2['checkin-booking'] < 0])

for i in range(len(data2['checkin-booking'])):
    if data2['checkin-booking'][i] < 0:
        data2['checkin-booking'][i] = 0


data2['checkout_date'] = pd.to_datetime(data2.checkout_date,dayfirst=True)

data2['checkout-checkin'] = (data2['checkout_date'] - data2['checkin_date']).dt.days

print(data2.loc[data2['checkout-checkin'] < 0])


####################################################

dataset2 = data2[['checkin-booking', 'checkout-checkin' ,'channel_code', 'main_product_code', 'numberofadults', 'numberofchildren', 'persontravellingid', 'resort_region_code', 'resort_type_code', 'room_type_booked_code', 'roomnights', 'season_holidayed_code', 'state_code_residence', 'state_code_resort','total_pax', 'member_age_buckets','booking_type_code', 'cluster_code', 'reservationstatusid_code', 'resort_id']].copy()

dataset2.fillna(0, inplace=True)

X2 = dataset2.iloc[:, :].values

X2[:, 15] = labelencoder_X_15.transform(X2[:, 15])

X2[:, 17] = labelencoder_X_17.transform(X2[:, 17])

X2[:, 18] = labelencoder_X_18.transform(X2[:, 18])

X2[:, 19] = labelencoder_X_19.transform(X2[:, 19])


#y_pred_testset = regressor.predict(X2)

y_pred_testset = xg_reg2.predict(X2)


# WRITING VALUES IN CSV FILE FOR SUBMISSION
from csv import DictReader

with open('test.csv') as f:
    resid_list = [row["reservation_id"] for row in DictReader(f)]
    
y_pred_testset_list = list(y_pred_testset)

#creating CSV file
df = pd.DataFrame(data={"reservation_id":resid_list , "amount_spent_per_room_night_scaled": y_pred_testset_list})
df.to_csv("submission5_xgboost_n250_subsample0.8maxdepth5.csv", sep=',',index=False)












