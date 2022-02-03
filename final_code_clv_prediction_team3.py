from __future__ import division
import pandas as pd
import numpy as np
from pandas.core import groupby
from pandas.io.pytables import dropna_doc
import pyarrow.parquet as pq
parquet_file = 'dataset/ml_dataset_train.parquet'
df1 = pd.read_parquet(parquet_file, engine = 'auto')
df2 = pd.read_parquet('dataset/ml_payers_extension.parquet', engine = 'auto')
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import joblib
import plydata.cat_tools as cat
import plotnine as pn
df = pd.concat([df1,df2])
import plotly.offline as pyoff
import plotly.graph_objs as go
from datetime import datetime, timedelta
from __future__ import division
import seaborn as sns 
import chart_studio.plotly as csp

pd.set_option('display.max_columns', None)

#### data preparation part 
# bool --> int 
for u in df.columns:
    if df[u].dtype == bool:
        df[u]=df[u].astype('int')

df = df.replace({False:0 , True:1})

# missing value rate  
total = df.isnull().sum().sort_values(ascending = False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis = 1, keys = ['total','percent'])

# drop features which have missing value rate more the 96%
dfall = df.dropna(thresh=len(df)*0.04,axis=1)

dfall = dfall.drop( columns = 'n_payments_package_key_ltv')

# Recheck Missing rate 
total_all = dfall.isnull().sum().sort_values(ascending = False)
percent_all = (dfall.isnull().sum()/dfall.isnull().count()).sort_values(ascending=False)
missing_data_all = pd.concat([total_all, percent_all], axis = 1, keys = ['total','percent'])

# drop masked_feature_26 因为缺失值比例大于key features
dfall.drop(columns = ['masked_feature_26'], inplace= True)

# other features with missing rate larger than 95% picked out by hand
dfall.drop(columns = ['nunique_iaps_bought'], inplace= True)
dfall.drop(columns = ['connected_fb'], inplace= True)
dfall.drop(columns = ['n_remove_ad_clicks'], inplace= True)
dfall.drop(columns = ['n_time_skip_buys'], inplace= True)
dfall.drop(columns = ['first_time_zone'], inplace= True)
dfall.drop(columns = ['nunique_countries'], inplace= True)
dfall.drop(columns = ['nunique_network_types'], inplace= True)

#mean best fit 
dfall['ram_max'] = dfall['ram_max'].fillna(dfall['ram_max'].mean())

#mean best fit
dfall['screen_density'] = dfall['screen_density'].fillna(dfall['screen_density'].mean())
dfall['screen_height'] = dfall['screen_height'].fillna(dfall['screen_height'].mean())
dfall['screen_width'] = dfall['screen_width'].fillna(dfall['screen_width'].mean())

#number of calender login days -> Used the minimum value of the column for every missing value, which is 1
dfall['n_calendar_login_days'] = dfall['n_calendar_login_days'].fillna(dfall['n_calendar_login_days'].min())

#number of sessions ended -> need to be at least 1
dfall['n_sessions_ended'] = dfall['n_sessions_ended'].fillna(dfall['n_sessions_ended'].min())

#also minimum 1
dfall['total_session_duration'] = dfall['total_session_duration'].fillna(dfall['total_session_duration'].min())

#same applies here
dfall['max_session_end_player_level'] = dfall['max_session_end_player_level'].fillna(dfall['max_session_end_player_level'].min())

#starts at 0
dfall['min_session_start_player_level'].fillna(0, inplace=True)

#take the latest value
dfall['bigmac_dollar_price'] = dfall['bigmac_dollar_price'].fillna(dfall['bigmac_dollar_price'].mode())

from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
dfall[['device_language','first_device_os','first_device_model','first_network_type','first_login_country','form_factor','first_login_weekday','first_login_day_time','most_frequent_country','most_frequent_network_type','first_device_manufacturer']] = enc.fit_transform(df[["device_language","first_device_os","first_device_model","first_network_type","first_login_country","form_factor","first_login_weekday","first_login_day_time","most_frequent_country","most_frequent_network_type","first_device_manufacturer"]])

# filling with 0 
dfall['n_payments_package_key'].fillna(0, inplace=True)
dfall['sum_payments_package_key'].fillna(0, inplace=True)
dfall['sum_payments_package_key_ltv'].fillna(0, inplace=True)
dfall['time_to_first_purchase'].fillna(0, inplace=True)
dfall['time_to_last_purchase'].fillna(0, inplace=True)
dfall['time_between_last_purchase_last_login'].fillna(0, inplace=True)
dfall['n_ads_watched'].fillna(0, inplace=True)
dfall['n_battlepass_lvls_finished'].fillna(0, inplace=True)
dfall['n_missions_completed'].fillna(0, inplace=True)
dfall['n_package_info_offers_viewed'].fillna(0, inplace=True)
dfall['n_package_tips_offers_viewed'].fillna(0, inplace=True)
dfall['n_viplevels_completed'].fillna(0, inplace=True)
dfall['masked_feature_0'].fillna(0, inplace=True)
dfall['masked_feature_1'].fillna(0, inplace=True)
dfall['masked_feature_2'].fillna(0, inplace=True)
dfall['masked_feature_3'].fillna(0, inplace=True)
dfall['masked_feature_4'].fillna(0, inplace=True)
dfall['masked_feature_5'].fillna(0, inplace=True)
dfall['masked_feature_6'].fillna(0, inplace=True)
dfall['masked_feature_40'].fillna(0, inplace=True)
dfall['masked_feature_41'].fillna(0, inplace=True)
dfall['n_ad_reward_claims'].fillna(0, inplace=True)
dfall['n_ad_reward_fails'].fillna(0, inplace=True)
dfall['masked_feature_7'].fillna(0, inplace=True)
dfall['masked_feature_8'].fillna(0, inplace=True)
dfall['masked_feature_13'].fillna(0, inplace=True)
dfall['masked_feature_14'].fillna(0, inplace=True)
dfall['masked_feature_15'].fillna(0, inplace=True)
dfall['masked_feature_20'].fillna(0, inplace=True)
dfall['masked_feature_21'].fillna(0, inplace=True)
dfall['masked_feature_22'].fillna(0, inplace=True)
dfall['masked_feature_23'].fillna(0, inplace=True)
dfall['masked_feature_24'].fillna(0, inplace=True)
dfall['masked_feature_25'].fillna(0, inplace=True)
dfall['masked_feature_29'].fillna(0, inplace=True)
dfall['connect_fb_attempt'].fillna(0, inplace=True)
dfall['n_ad_reward_fails'].fillna(0, inplace=True)
dfall['n_ad_reward_claims'].fillna(0, inplace=True)
dfall['masked_feature_20'].fillna(0, inplace=True)
dfall['masked_feature_21'].fillna(0, inplace=True)
dfall['masked_feature_22'].fillna(0, inplace=True)
dfall['masked_feature_23'].fillna(0, inplace=True)
dfall['masked_feature_24'].fillna(0, inplace=True)
dfall['masked_feature_25'].fillna(0, inplace=True)

dfall.fillna(0, inplace= True)

dfall.set_index('account_id', inplace = True)
dfall['masked_feature_40'] = dfall.masked_feature_40.astype(float)
dfall

#### lasso regression part 

df = dfall.copy()

df = df.fillna(0)

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, train_test_split, GridSearchCV
X = df.drop(columns='sum_payments_package_key_ltv')
y = df[['sum_payments_package_key_ltv']]
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, LogisticRegression,LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from sklearn import metrics


# pipeline 
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Lasso())
])

search= GridSearchCV(pipeline,
    {'model__alpha':np.arange(0.1,10,0.5)},
    cv =3,
scoring = "neg_mean_squared_error", verbose=3
)

search.fit(X_train,y_train)

search.best_params_

coefficients = search.best_estimator_.named_steps['model'].coef_

importance = np.abs(coefficients)


# lasso the alternatvie
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

lassoreg = Lasso(alpha = 0.6)

lassoreg.fit(X_train_std,y_train)

print('R squared training set', round(lassoreg.score(X_train_std,y_train)*100, 2))
print('R squared test set', round(lassoreg.score(X_test_std,y_test)*100, 2))
print('training accuracy:', lassoreg.score(X_train_std, y_train))
print('test accuracy:', lassoreg.score(X_test_std, y_test))

# Lasso selected features 
lassoreg.intercept_
lassoreg.coef_


from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

## results 
#training set
pred_y = lassoreg.predict(X_train_std)
mse_ = mean_squared_error(y_train, pred_y)
print('MSE training set', round(mse_, 2))
rmse_= sqrt(mean_squared_error(y_train, pred_y))
rmse_

#testing set 
pred_y_test = lassoreg.predict(X_test_std)
mse_test = mean_squared_error(y_test, pred_y_test)
print('MSE testing set', round(mse_test, 2))
rmse_test = sqrt(mean_squared_error(y_test, pred_y_test))
rmse_test

## measure error figure test set 
fig, ax = plt.subplots()
ax.scatter(y_test, pred_y_test)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured_lasso')
ax.set_ylabel('Predicted_lasso')
plt.show()

plt.style.use('seaborn')
fig = plt.figure(figsize=(13,7))
ax = fig.add_subplot(1,1,1)

arr1 = plt.scatter(X_test_std[:,21], y_test , c='red', label = 'real')
arr2 = plt.scatter(X_test_std[:,21], pred_y_test, c='green', label = 'lasso')


# roll of alphas 
alphas = np.arange(0.1,2,0.5)
lasso = Lasso(max_iter=100000000)
coefs = []

for a in alphas: 
    lasso.set_params(alpha = a)
    lasso.fit(X_train_std,y_train)
    coefs.append(lasso.coef_)


## figure of alphas
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('standardized Coefficients')
plt.title('Lasso coefficients as a function of alpha')
plt.plot(alphas, coefs)


##### multiple linear regression part 

from __future__ import division
from io import DEFAULT_BUFFER_SIZE
import pandas as pd
import numpy as np
from pandas.core import groupby
from pandas.io.pytables import dropna_doc
from pandas.tseries.offsets import YearBegin
import pyarrow.parquet as pq
df_linear_corr = dfall.copy()
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import joblib
import plydata.cat_tools as cat
import plotnine as pn
import plotly.offline as pyoff
import plotly.graph_objs as go

df_linear_corr = dfall[['masked_feature_15','masked_feature_24','n_payments_package_key','sum_payments_package_key','masked_feature_41','battlepass_0.0','battlepass_22.0','battlepass_23.0','time_to_last_purchase','sum_payments_package_key_ltv']]
pd.set_option('display.max_columns', None)


# DataFrame creation
df_LR = df_linear_corr

df_LR_RFM = df_linear_corr[['n_payments_package_key','sum_payments_package_key','time_to_last_purchase','sum_payments_package_key_ltv']]

df_LR_withoutRFM = df_linear_corr[['masked_feature_15','masked_feature_24','masked_feature_41','battlepass_0.0','battlepass_22.0','battlepass_23.0','sum_payments_package_key_ltv']]

# LR 
from sklearn.model_selection import train_test_split
X = df_LR[['masked_feature_15','masked_feature_24','n_payments_package_key','sum_payments_package_key','masked_feature_41','battlepass_0.0','battlepass_22.0','battlepass_23.0','time_to_last_purchase']]
y = df_LR[['sum_payments_package_key_ltv']]
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=41)

## standardization
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf 
import seaborn as sns
import numpy as np
import patsy
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

## linear regression 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, r2_score, explained_variance_score as EVS, mean_squared_error as MSE

Lreg = LinearRegression()

formula = 'y_train~X_train'
model = smf.ols(formula, data={'y_train':y_train,'X_train':X_scaled})
result = model.fit()
result.summary()
Lreg.fit(X_scaled,y_train)
y_pred_scaled = Lreg.predict(X_test_scaled)
metrics.mean_squared_error(y_test,y_pred_scaled)
np.sqrt(metrics.mean_squared_error(y_test, y_pred_scaled)) 
r2_score(y_test, y_pred_scaled)
LRresult_intercept = Lreg.intercept_
LRresult_coef = Lreg.coef_

##training set result
from sklearn.metrics import mean_squared_error
from math import sqrt
pred_y = Lreg.predict(X_scaled)
mse_ = mean_squared_error(y_train, pred_y)
print('MSE training set', round(mse_, 2))
rmse_= sqrt(mean_squared_error(y_train, pred_y))
r2_score(y_train, pred_y)

## cross validation 
from sklearn.model_selection import cross_val_predict, cross_val_score
predicted = cross_val_predict(Lreg, X_scaled, y_train, cv = 10)
metrics.mean_squared_error(y_train, predicted)
np.sqrt(metrics.mean_squared_error(y_train, predicted))
r2_score(y_train, predicted)

score = cross_val_score(Lreg,X_train,y_train,cv=10)
print('交叉验证平均得分:{:.3f}'.format(score.mean()))

# figure
fig, ax = plt.subplots()
ax.scatter(y_train, predicted)
ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

# LR only with RFM 
X_RFM = df_LR_RFM[['time_to_last_purchase','n_payments_package_key','sum_payments_package_key']]
y_RFM = df_LR_RFM[['sum_payments_package_key_ltv']]
X_RFMtrain, X_RFMtest, y_RFMtrain, y_RFMtest = train_test_split(X_RFM,y_RFM,random_state=41)

## standardization
scaler = StandardScaler()
X_RFM_scaled = scaler.fit_transform(X_RFMtrain)
X_RFM_test_scaled = scaler.fit_transform(X_RFMtest)

## regression
Lreg.fit(X_RFM_scaled, y_RFMtrain)
Lreg.intercept_
Lreg.coef_

y_RFMpred = Lreg.predict(X_RFM_test_scaled)

metrics.mean_squared_error(y_RFMtest,y_RFMpred)
np.sqrt(metrics.mean_squared_error(y_RFMtest, y_RFMpred))
r2_score(y_RFMtest, y_RFMpred)

# LR without RFM 
X_0RFM = df_LR_withoutRFM[['masked_feature_15','masked_feature_24','masked_feature_41','battlepass_0.0','battlepass_22.0','battlepass_23.0']]
y_0RFM = df_LR_withoutRFM[['sum_payments_package_key_ltv']]
X_0RFMtrain, X_0RFMtest, y_0RFMtrain, y_0RFMtest = train_test_split(X_0RFM,y_0RFM,random_state=1)

## standardization
scaler = StandardScaler()
X_0RFM_scaled = scaler.fit_transform(X_0RFMtrain)
X_0RFM_test_scaled = scaler.fit_transform(X_0RFMtest)

## regression
Lreg.fit(X_0RFM_scaled, y_0RFMtrain)
Lreg.intercept_
Lreg.coef_

y_0RFMpred = Lreg.predict(X_0RFM_test_scaled)

metrics.mean_squared_error(y_0RFMtest,y_0RFMpred)
np.sqrt(metrics.mean_squared_error(y_0RFMtest, y_0RFMpred))
r2_score(y_0RFMtest, y_0RFMpred)

