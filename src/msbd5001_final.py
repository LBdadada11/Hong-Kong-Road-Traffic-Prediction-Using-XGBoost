#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author : Priyanko Basuchaudhuri
# Date : 12-Dec-2020
# Version : 1.0

# Import libraries

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.utils import shuffle

df = pd.read_csv('../input/msbd5001-fall2020/train.csv')  # read in csv file as a DataFrame
df[['date']] = df[['date']].apply(pd.to_datetime, dayfirst=True)  # convert to datetime , Note day is coming first


# Function to separate day of the week , month , year

def extract_date(df, column):
    df[column + '_year'] = df[column].apply(lambda x: x.year)
    df[column + '_month'] = df[column].apply(lambda x: x.month)
    df[column + '_hour'] = df[column].apply(lambda x: x.hour)
    df[column + '_minute'] = df[column].apply(lambda x: x.minute)
    df[column + '_monthday'] = df[column].apply(lambda x: x.day)
    df[column + '_weekday'] = df[column].apply(lambda x: x.weekday())


# Function to separate day of the week , month , year - retaining the cyclical pattern in encoded value

from datetime import datetime
from math import sin, cos, pi
days_in_month = [
    31,
    28,
    31,
    30,
    31,
    30,
    31,
    31,
    30,
    31,
    30,
    31,
    ]


def extract_cycles(df, column):
    '''
    Get the cyclic properties of a datetime,
    represented as points on the unit circle.
    Arguments
    ---------
    df : datetime object
    '''

    df[column + '_sin_month'] = df[column].apply(lambda x: \
            np.sin((x.month - 1) / 12) * 2 * np.pi)
    df[column + '_sin_day'] = df[column].apply(lambda x: np.sin((x.day
            - 1) / days_in_month[x.month - 1]) * 2 * np.pi)
    df[column + '_sin_weekday'] = df[column].apply(lambda x: \
            np.sin(x.weekday() / 7) * 2 * np.pi)
    df[column + '_sin_hour'] = df[column].apply(lambda x: np.sin(x.hour
            / 24) * 2 * np.pi)
    df[column + '_cos_month'] = df[column].apply(lambda x: \
            np.cos((x.month - 1) / 12) * 2 * np.pi)
    df[column + '_cos_day'] = df[column].apply(lambda x: np.cos((x.day
            - 1) / days_in_month[x.month - 1]) * 2 * np.pi)
    df[column + '_cos_weekday'] = df[column].apply(lambda x: \
            np.cos(x.weekday() / 7) * 2 * np.pi)
    df[column + '_cos_hour'] = df[column].apply(lambda x: np.cos(x.hour
            / 24) * 2 * np.pi)


# Extract additional features from date

extract_date(df, 'date')
extract_cycles(df, 'date')

df.dtypes  # After expanding features

df.date_minute.unique()

# As no value in date_minute - delete the column

del df['date_minute']

# check if data missing

for c in df.columns:
    num_na = df[c].count()
    if num_na > 0:
        print round(num_na / float(len(df)), 3)

df['speed'].isna().sum()  # Check if speed data missing

# Simple Plot

plt.figure(figsize=(20, 5))
plt.plot(df.index, df['speed'])
plt.title('Speed')
plt.xlabel('Observation Index')
plt.ylabel('Speed')
plt.show(block=False)

# Year to Year Avg Speed

avgAnnualData = df[['speed']]
avgAnnualData['Year'] = df['date_year']
avgAnnualData = avgAnnualData.groupby('Year', as_index=False).mean()

plt.figure(figsize=(10, 5))
plt.bar(avgAnnualData['Year'], avgAnnualData['speed'], width=0.4)
plt.title('Average Speed for each Year')
plt.xlabel('Year')
plt.ylabel('Average Speed')
plt.show(block=False)

# Month to Month Avg Speed

avgMonthlyData = df[['speed']]
avgMonthlyData['Month'] = df['date_month']
avgMonthlyData = avgMonthlyData.groupby('Month', as_index=False).mean()

plt.figure(figsize=(10, 5))
plt.bar(avgMonthlyData['Month'], avgMonthlyData['speed'], width=0.4)
plt.title('Average Speed for each Month')
plt.xlabel('Month')
plt.ylabel('Average Speed')
plt.show(block=False)

# Average Speed During each Month Year by Year

yearList = list(df.date_year.unique())
print yearList

monthYearWiseAvgTPData = df[['date_month', 'speed']]
monthYearWiseAvgTPData['Year'] = df['date_year']
monthYearWiseAvgTPData = monthYearWiseAvgTPData.groupby(['Year',
        'date_month'], as_index=False).mean()
fig = plt.figure(figsize=(20, 20))
fig.tight_layout()
for i in range(len(yearList)):
    dataYearly = monthYearWiseAvgTPData[monthYearWiseAvgTPData['Year']
            == yearList[i]]  # select dataframe with month = i
    ax = fig.add_subplot(4, 2, i + 1)  # add subplot in the i-th position on a grid 4 * 2
    ax.title.set_text('Average speed during each month for Year '
                      + str(yearList[i]))
    ax.plot(dataYearly['date_month'], dataYearly['speed'])

# Average speed Hour to Hour in a day

avgTimePeriodData = df[['date_hour', 'speed']]
avgTimePeriodData = avgTimePeriodData.groupby('date_hour',
        as_index=False).mean()

plt.figure(figsize=(20, 5))
plt.bar(avgTimePeriodData['date_hour'], avgTimePeriodData['speed'])

# plt.xticks(np.arange(0,96), timeList, rotation='vertical')

plt.title('Average Vehicle Flow for each TimePeriod of the day')
plt.xlabel('Hour Of the Day')
plt.ylabel('Average Speed')
plt.show(block=False)

# Average Speed variation during each weekday - day to day

dayList = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    ]
weekDayWiseAvgTPData = df[['date_hour', 'speed']]
weekDayWiseAvgTPData['Day'] = df['date_weekday']
weekDayWiseAvgTPData = weekDayWiseAvgTPData.groupby(['Day', 'date_hour'
        ], as_index=False).mean()
fig = plt.figure(figsize=(20, 20))
fig.tight_layout()
for i in range(7):
    dataWeekly = weekDayWiseAvgTPData[weekDayWiseAvgTPData['Day']
            == dayList[i]]  # select dataframe with month = i
    ax = fig.add_subplot(4, 2, i + 1)  # add subplot in the i-th position on a grid 4 * 2
    ax.title.set_text('Average speed variation on the day for weekday '
                      + str(dayList[i]))
    ax.plot(dataWeekly['date_hour'], dataWeekly['speed'])

# Average Speed variation for each Day of Week

dayList = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    ]

avgDayOfWeekData = df[['speed']]
avgDayOfWeekData['Day'] = df['date_weekday']
avgDayOfWeekData = avgDayOfWeekData.groupby(['Day'],
        as_index=False).mean()
plt.figure(figsize=(10, 5))
plt.plot(avgDayOfWeekData['Day'], avgDayOfWeekData['speed'])
plt.scatter(avgDayOfWeekData['Day'], avgDayOfWeekData['speed'],
            c='orange')
plt.xticks(np.arange(0, 7), dayList, rotation='horizontal')
plt.title('Average Speed for each Day of Week')
plt.xlabel('Day')
plt.ylabel('Average Speed')
plt.show(block=False)

# Average speed variation of Vehicles Month to Month

monthNumList = list(range(1, 13))
monthList = [
    'January',
    'February',
    'March',
    'April',
    'May',
    'June',
    'July',
    'August',
    'September',
    'October',
    'November',
    'December',
    ]

mthWiseAvgDayOfWeekData = df[['speed']]
mthWiseAvgDayOfWeekData['Day'] = df['date_weekday']
mthWiseAvgDayOfWeekData['Month'] = df['date_month']
mthWiseAvgDayOfWeekData = mthWiseAvgDayOfWeekData.groupby(['Month',
        'Day'], as_index=False).mean()

fig = plt.figure(figsize=(20, 15))
fig.tight_layout()
for i in range(12):
    dataMonthly = \
        mthWiseAvgDayOfWeekData[mthWiseAvgDayOfWeekData['Month']
                                == monthNumList[i]]  # select dataframe with month = i

    # print(dataMonthly)

    ax = fig.add_subplot(4, 3, i + 1)  # add subplot in the i-th position on a grid 12x1
    ax.title.set_text('Average speed variation of Vehicles '
                      + monthList[i])
    ax.plot(dataMonthly['Day'], dataMonthly['speed'])
    ax.scatter(dataMonthly['Day'], dataMonthly['speed'], c='orange')
    ax.set_xticks(dataMonthly['Day'].unique())  # set x axis

# Average speed of Vehicles - each month day - Month to Month

monthNumList = list(range(1, 13))
monthList = [
    'January',
    'February',
    'March',
    'April',
    'May',
    'June',
    'July',
    'August',
    'September',
    'October',
    'November',
    'December',
    ]

mthWiseAvgDayOfWeekData = df[['speed']]
mthWiseAvgDayOfWeekData['Month Day'] = df['date_monthday']
mthWiseAvgDayOfWeekData['Month'] = df['date_month']
mthWiseAvgDayOfWeekData = mthWiseAvgDayOfWeekData.groupby(['Month',
        'Month Day'], as_index=False).mean()

fig = plt.figure(figsize=(20, 15))
fig.tight_layout()
for i in range(12):
    dataMonthly = \
        mthWiseAvgDayOfWeekData[mthWiseAvgDayOfWeekData['Month']
                                == monthNumList[i]]  # select dataframe with month = i

    # print(dataMonthly)

    ax = fig.add_subplot(4, 3, i + 1)  # add subplot in the i-th position on a grid 12x1
    ax.title.set_text('Average speed of Vehicles ' + monthList[i])
    ax.plot(dataMonthly['Month Day'], dataMonthly['speed'])
    ax.scatter(dataMonthly['Month Day'], dataMonthly['speed'],
               c='orange')
    ax.set_xticks(dataMonthly['Month Day'].unique())  # set x axis

# Most important trend occurs within the hours of each day , within days of each week - and
# some trend exists within weekdays of each month. Average speed in each day of a month
# varies but does not have a consistent trend.

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

original_cols = list(df.columns)
cols = [
    'id',
    'date',
    'date_year',
    'date_month',
    'date_hour',
    'date_monthday',
    'date_weekday',
    'date_sin_month',
    'date_sin_day',
    'date_sin_weekday',
    'date_sin_hour',
    'date_cos_month',
    'date_cos_day',
    'date_cos_weekday',
    'date_cos_hour',
    'speed',
    ]
df = df[cols]
df = shuffle(df, random_state=1234)
X = df.drop(['speed', 'date', 'id'], axis=1)
y = df['speed']
(X_train, X_test, y_train, y_test) = train_test_split(X, y,
        test_size=0.2, random_state=1)
print len(X_train)
print len(X_test)

X_train_nonen = X_train
X_test_nonen = X_test

from xgboost import XGBRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, GridSearchCV, \
    KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import ensemble

# A parameter grid for XGBoost

paramGrid = {
    'colsample_bytree': [0.8],
    'max_depth': [8, 9, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [1750, 2000],
    'subsample': [0.8],
    }

# Initialize XGBRegressor

regXGB = XGBRegressor()

rsc = GridSearchCV(estimator=regXGB, param_grid=paramGrid, scoring='r2'
                   , cv=10)
grid_result = rsc.fit(X_train_nonen, y_train)

# Choose best parametrs from the Grid

best_params = grid_result.best_params_
print grid_result.best_params_

# Use best parameters in Model

regXGB = XGBRegressor(**grid_result.best_params_)

# Train the model

regXGB.fit(X_train_nonen, y_train)
y_predicted = regXGB.predict(X_test_nonen)

# Print result metrics

print 'R2 Score %.3f' % r2_score(y_test, y_predicted)
print 'Mean Squared error %.2f' % mean_squared_error(y_test,
        y_predicted)

# Plot result against ground truth

(fig, ax) = plt.subplots()
ax.scatter(y_test, y_predicted, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Ground Truth vs Predicted')
plt.show()

# Transform the test data

df_test = pd.read_csv('../input/msbd5001-fall2020/test.csv')  # read in csv file as a DataFrame
df_test[['date']] = df_test[['date']].apply(pd.to_datetime,
        dayfirst=True)  # convert to datetime
extract_date(df_test, 'date')
extract_date(df_test, 'date')
extract_cycles(df_test, 'date')
del df_test['date_minute']

df_test.head()

cols = [
    'id',
    'date',
    'date_year',
    'date_month',
    'date_hour',
    'date_monthday',
    'date_weekday',
    'date_sin_month',
    'date_sin_day',
    'date_sin_weekday',
    'date_sin_hour',
    'date_cos_month',
    'date_cos_day',
    'date_cos_weekday',
    'date_cos_hour',
    ]
df_test = df_test[cols]
X_out = df_test.drop(['date', 'id'], axis=1)
X_out_nonen = X_out

# Predit from the test dataset

speed_output = regXGB.predict(X_out_nonen)
print speed_output

# Prepare submission file

my_submission = pd.DataFrame({'id': df_test.id, 'speed': speed_output})
my_submission.to_csv('priyanko_final_submission.csv', index=False)
