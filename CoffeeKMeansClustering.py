# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 00:23:37 2020

@author: Ashish
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import preprocessing

sns.set()
data=pd.read_csv(r'A:\Ashish\TestingProjects\Datasets\coffee-quality-database-from-cqi\merged_data_cleaned.csv',\
                 index_col=0)
"""
1) Clustering the data into different types of coffee (Branding)
    Similar type of coffee can be acquired from different farms.
2) Classifying inward data into these clusters
3) Predicting the coffee grade according to climate,location.
    """

"""1) CLUSTERING """
branding=['Species','Aroma','Flavor','Aftertaste','Acidity','Body','Balance','Uniformity','Sweetness','Cupper.Points','Moisture','Color']
clusterinfo=data[branding]
clusterinfo=clusterinfo[clusterinfo['Flavor']!=0]

clusterinfo=clusterinfo[clusterinfo['Aroma']>=6.4]
clusterinfo=clusterinfo[clusterinfo['Acidity']>=6.5]
clusterinfo=clusterinfo[clusterinfo['Uniformity']>=6.5]
clusterinfo=clusterinfo[clusterinfo['Sweetness']>6]
clusterinfo=clusterinfo[clusterinfo['Balance']>=6.4]
clusterinfo=clusterinfo[clusterinfo['Body']>=6]
clusterinfo=clusterinfo[clusterinfo['Moisture']<=0.25]

clusterinfo['Species'].unique()
clusterinfo['Species']=clusterinfo['Species'].map({'Arabica':1,'Robusta':0})    

clusterinfo['Color'].fillna('None',inplace=True)
clusterinfo['Color'].unique()

clusterinfo['Color']=clusterinfo['Color'].map({'Green':1,'Bluish-Green':2,'None':0,'Blue-Green':3})    

datascaled=preprocessing.scale(clusterinfo)

wcss=[]
for i in range(1,9):
    kmeans1=KMeans(i)
    kmeans1.fit(datascaled)
    wcss_iter=kmeans1.inertia_
    wcss.append(wcss_iter)

number_clusters=range(1,9)
plt.plot(number_clusters,wcss)
plt.xlabel('No of clusters')
plt.ylabel('WCSS')

kmeans=KMeans(4)
kmeans.fit(datascaled)

clusterdata=kmeans.fit_predict(datascaled)
#plt.scatter(clusterinfo['Moisture'],clusterinfo['Flavor'])
plt.scatter(clusterinfo['Aroma'],clusterinfo['Flavor'],c=clusterdata,cmap='rainbow')

clusterinfo['Clusters']=clusterdata
""" END 1) """