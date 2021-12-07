#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pyarrow


# In[2]:


pip install plotly


# In[3]:


pip install chart-studio


# In[4]:


from __future__ import division
import pandas as pd
import numpy as np
from pandas.core import groupby
from pandas.io.pytables import dropna_doc
import pyarrow.parquet as pq
df1 = pd.read_parquet('ml_dataset_train.parquet', engine = 'auto') # given complete dataset
df2 = pd.read_parquet('transactional_dataset_train.parquet', engine = 'auto') # only transactional dataset
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import joblib
df_3 = pd.merge(left=df1,right=df2,on='account_id') # merged set for transactions
import plotly.offline as pyoff
import plotly.graph_objs as go
from datetime import datetime, timedelta
import seaborn as sns 
import chart_studio.plotly as csp
pyoff.init_notebook_mode()


# In[5]:


df_3['event_time'] = pd.to_datetime(df_3['event_time']) #convert the string date field to datetime


# In[6]:


#recency
df_4 = pd.DataFrame(df_3['account_id'].unique()) #create a generic user dataframe to keep AccountID and new segmentation scores
df_4.columns = ['account_id']

#get the max purchase date for each customer and create a dataframe with it
df_max_purchase = df_3.groupby('account_id').event_time.max().reset_index()
df_max_purchase.columns = ['account_id','MaxPurchaseDate']

#we take our observation point as the max invoice date in our dataset
df_max_purchase['Recency'] = (df_max_purchase['MaxPurchaseDate'].max() - df_max_purchase['MaxPurchaseDate']).dt.days

#merge this dataframe to our new user dataframe
df_4 = pd.merge(df_4, df_max_purchase[['account_id','Recency']], on='account_id')

df_4.head()

#plot a recency histogram

plot_data = [
    go.Histogram(
        x=df_4['Recency']
    )
]

plot_layout = go.Layout(
        title='Recency'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[7]:


df_4.Recency.describe()


# In[8]:


from sklearn.cluster import KMeans #Kmeans clustering to find adequate number of clusters

sse={}
df_recency = df_4[['Recency']]
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df_recency)
    df_recency["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_ 
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.show() #displays number of ideal clusters


# In[9]:


#build 4 clusters for recency and add it to dataframe
kmeans = KMeans(n_clusters=4)
kmeans.fit(df_4[['Recency']])
df_4['RecencyCluster'] = kmeans.predict(df_4[['Recency']])

#function for ordering cluster numbers
def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final

df_4 = order_cluster('RecencyCluster', 'Recency',df_4,False)


# In[10]:


df_4.groupby('RecencyCluster')['Recency'].describe() # Number 3 is the priority group, as they will spend money after approximately 17 days again


# In[11]:


#frequency
#get order counts for each user and create a dataframe with it
df_frequency = df_3.groupby('account_id').event_time.count().reset_index()
df_frequency.columns = ['account_id','Frequency']

#add this data to our main dataframe
df_4 = pd.merge(df_4, df_frequency, on='account_id')

#plot the histogram
plot_data = [
    go.Histogram(
        x=df_4.query('Frequency < 50')['Frequency']
    )
]

plot_layout = go.Layout(
        title='Frequency'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[12]:


#k-means
kmeans = KMeans(n_clusters=4)
kmeans.fit(df_4[['Frequency']])
df_4['FrequencyCluster'] = kmeans.predict(df_4[['Frequency']])

#order the frequency cluster
df_4 = order_cluster('FrequencyCluster', 'Frequency',df_4,True)

#see details of each cluster
df_4.groupby('FrequencyCluster')['Frequency'].describe()


# In[13]:


#Monetary Value
#calculate revenue for each customer
df_3['Revenue'] = df_3['amount']
df_revenue = df_3.groupby('account_id').Revenue.sum().reset_index()

#merge it with our main dataframe
df_4 = pd.merge(df_4, df_revenue, on='account_id')

#plot the histogram
plot_data = [
    go.Histogram(
        x=df_4.query('Revenue < 200')['Revenue']
    )
]

plot_layout = go.Layout(
        title='Monetary Value'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[14]:


df_4.Revenue.describe()


# In[15]:


#apply clustering
kmeans = KMeans(n_clusters=4)
kmeans.fit(df_4[['Revenue']])
df_4['RevenueCluster'] = kmeans.predict(df_4[['Revenue']])


#order the cluster numbers
df_4 = order_cluster('RevenueCluster', 'Revenue',df_4,True)

#show details of the dataframe
df_4.groupby('RevenueCluster')['Revenue'].describe()


# In[16]:


#calculate overall score and use mean() to see details
df_4['OverallScore'] = df_4['RecencyCluster'] + df_4['FrequencyCluster'] + df_4['RevenueCluster']
df_4.groupby('OverallScore')['Recency','Frequency','Revenue'].mean()
df_4['Segment'] = 'Low-Value'
df_4.loc[df_4['OverallScore']>2,'Segment'] = 'Mid-Value' 
df_4.loc[df_4['OverallScore']>4,'Segment'] = 'High-Value' 


# In[17]:


df_4


# In[ ]:




