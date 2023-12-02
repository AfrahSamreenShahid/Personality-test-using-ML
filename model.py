import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import warnings
warnings.filterwarnings('ignore')
data_raw= pd.read_csv('big5ds.csv',sep='\t')
data = data_raw.copy()
data.drop(data.columns[50:107], axis=1, inplace=True)
data.drop(data.columns[50:], axis=1, inplace=True)
data.head(10)
print('Is there any missing value? ', data.isnull().values.any())
print('How many missing values? ', data.isnull().values.sum())
data.dropna(inplace=True)
print('Number of participants after eliminating missing values: ', len(data))
# For ease of calculation lets scale all the values between 0-1 and take a sample of 5000 
from sklearn.preprocessing import MinMaxScaler
columns = list(data.columns)
# For ease of calculation lets scale all the values between 0-1 and take a sample of 5000
#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler(feature_range=(0,1))
#df = scaler.fit_transform(data)
#df = pd.DataFrame(df, columns=columns)
df_sample = data[:5000]
# In[6]:
# Visualize the elbow
from sklearn.cluster import KMeans
#from yellowbrick.cluster import KElbowVisualizer
kmeans = KMeans()
#visualizer = KElbowVisualizer(kmeans, k=(2,15))
#visualizer.fit(df_sample)
#visualizer.poof()
# In[7]:
# Creating K-means Cluster Model
from sklearn.cluster import KMeans
# I define 5 clusters and fit my model
kmeans = KMeans(n_clusters=5)
k_fit = kmeans.fit(data_model)
# In[8]:
# Predicting the Clusters
pd.options.display.max_columns = 50
predictions = k_fit.labels_
data_model['Clusters'] = predictions
data_model.head(10)
# In[9]:
data_model.Clusters.value_counts()
# In[10]:
# Summing up the different questions groups
col_list = list(data_model)
ext = col_list[0:10]
est = col_list[10:20]
agr = col_list[20:30]
csn = col_list[30:40]
opn = col_list[40:50]

data_sums = pd.DataFrame()
data_sums['extroversion'] = data_model[ext].sum(axis=1)/10
data_sums['neurotic'] = data_model[est].sum(axis=1)/10
data_sums['agreeable'] = data_model[agr].sum(axis=1)/10
data_sums['conscientious'] = data_model[csn].sum(axis=1)/10
data_sums['open'] = data_model[opn].sum(axis=1)/10
data_sums['clusters'] = predictions
data_sums.groupby('clusters').mean()


# In[11]:


# Visualizing the means for each cluster
dataclusters = data_sums.groupby('clusters').mean()
plt.figure(figsize=(20,3))
for i in range(0, 5):
    plt.subplot(1,5,i+1)
    plt.bar(dataclusters.columns, dataclusters.iloc[:, i], color='green', alpha=0.2)
    plt.plot(dataclusters.columns, dataclusters.iloc[:, i], color='red')
    plt.title('Cluster ' + str(i))
    plt.xticks(rotation=45)
    plt.ylim(0,4);


# <H1><B>Implementing the Model to See My Personality</B></H1>

# In[12]:


my_data = pd.read_excel('AFRAH BIGFIVEDS.xlsx')
my_data


# In[13]:


my_personality = k_fit.predict(my_data)
print('My Personality Cluster: ', my_personality)


# In[16]:


joblib.dump(k_fit, "train_model.pkl")





