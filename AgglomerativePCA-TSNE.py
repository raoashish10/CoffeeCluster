# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 09:51:26 2020

@author: Ashish
"""

import scipy.cluster.hierarchy as shc

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data=pd.read_csv(r'A:\Ashish\TestingProjects\Datasets\coffee-quality-database-from-cqi\merged_data_cleaned.csv',\
                 index_col=0)

branding=['Aroma','Flavor','Aftertaste','Acidity','Body','Balance','Uniformity','Sweetness','Cupper.Points','Moisture','Color']

#branding=['Species','Aroma','Flavor','Aftertaste','Acidity','Body','Balance','Uniformity','Sweetness','Cupper.Points','Moisture','Color']
clusterinfo=data[branding]

#***Cleaning data

clusterinfo=clusterinfo[clusterinfo['Flavor']>6.5]
clusterinfo=clusterinfo[clusterinfo['Aroma']>=6.4]
clusterinfo=clusterinfo[clusterinfo['Acidity']>=6.5]
clusterinfo=clusterinfo[clusterinfo['Uniformity']>=6.5]
clusterinfo=clusterinfo[clusterinfo['Sweetness']>6]
clusterinfo=clusterinfo[clusterinfo['Balance']>=6.4]
clusterinfo=clusterinfo[clusterinfo['Body']>=6]
clusterinfo=clusterinfo[clusterinfo['Moisture']<=0.25]
clusterinfo=clusterinfo[clusterinfo['Aftertaste']>6.5]

data=data[data['Flavor']>6.5]
data=data[data['Aroma']>=6.4]
data=data[data['Acidity']>=6.5]
data=data[data['Uniformity']>=6.5]
data=data[data['Sweetness']>6]
data=data[data['Balance']>=6.4]
data=data[data['Body']>=6]
data=data[data['Moisture']<=0.25]
data=data[data['Aftertaste']>6.5]

clusterinfo['Color'].fillna('None',inplace=True)
clusterinfo['Color'].unique()

clusterinfo['Color']=clusterinfo['Color'].map({'Green':1,'Bluish-Green':2,'None':0,'Blue-Green':3})    

X_std = StandardScaler().fit_transform(clusterinfo)

#***Cleaning data END

#***PCA

#Method 1 (Checking variance)
pca = PCA()

pca.fit(X_std)
#The amount of variance that each PC explains
var= pca.explained_variance_ratio_

#Cumulative Variance explains
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

print(var1)
plt.plot(var1)
#Method 1 ends 

#Method 2 (Finding Elbow)

pca = PCA()
principalComponents = pca.fit_transform(X_std)

#Checking variance by each component
pca.explained_variance_ratio_
# 57% of the variance is contributed by the first 2 components 

features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)
# Method 2 ends

#Choosing 5 components, more than 80% Variance +  very small elbow if we ignore the first big one.
pca = PCA(n_components=5)
pc = pca.fit_transform(X_std)

#Clustering using Agglomerative

clustering = AgglomerativeClustering(n_clusters = 10)
clustering.fit(pc)

labels=clustering.labels_

clusterinfo['Labels']=labels
data['Labels']=labels
data.to_csv(r'A:\Ashish\TestingProjects\Datasets\coffee-quality-database-from-cqi\coffeecluster.csv',index=False)

#TSNE for visualising the clusters. 
tsne=TSNE()
X_embedded=tsne.fit_transform(pc)

palette = sns.color_palette("bright", clustering.n_clusters)
sns.scatterplot(X_embedded[:,0], X_embedded[:,1],hue=labels, palette=palette,legend=False)
plt.title('Agglomerative  n={}  comp={}'.format(clustering.n_clusters,pca.n_components))

#Dendrogram

plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(pc, method='ward'))

print("Starting Classification Procedure:")

while(True):
    Aroma=float(input('Enter Aroma Rating: '))
    if Aroma <10 and Aroma>0:
        break
    else:
        print('Wrong Input.')
while(True):    
    Flavor=float(input('Enter Flavor Rating: '))
    if Flavor<10 and Flavor>0:
        break
    else:
        print('Wrong Input.')

while(True):
    Aftertaste=float(input('Enter Aftertaste Rating: '))
    if Aftertaste<10 and Aftertaste>0:
        break
    else:
        print('Wrong Input.')

while(True):
    Acidity=float(input('Enter Acidity Rating: '))
    if Acidity < 10 and Acidity >0:
        break
    else:
        print('Wrong Input.')
        

while(True):
    Body=float(input('Enter Body Rating: '))
    if Body < 10 and Body >0:
        break
    else:
        print('Wrong Input.')

while(True):
    Balance=float(input('Enter Balance Rating: '))
    if Balance < 10 and Balance >0:
        break
    else:
        print('Wrong Input.')

while(True):
    Uniformity=float(input('Enter Uniformity Rating: '))
    if Uniformity < 10 and Uniformity >0:
        break
    else:
        print('Wrong Input.')

while(True):
    Sweetness=float(input('Enter Sweetness Rating: '))
    if Sweetness < 10 and Sweetness >0:
        break
    else:
        print('Wrong Input.')

while(True):
    CupperPoints=float(input('Enter Cupper Points: '))
    if CupperPoints < 10 and CupperPoints >0:
        break
    else:
        print('Wrong Input.')

while(True):
    Moisture=float(input('Enter Moisture Rating: '))
    if Moisture < 10 and Moisture >0:
        break
    else:
        print('Wrong Input.')

Color=input('Enter Color : ')
if Color=='Green':
    Color=1
elif Color=='Bluish-Green':
    Color=2
elif Color=='Blue-Green':
    Color=3
elif 'None':
    Color=0
else:
    Color=4
        
data=pd.read_csv(r'A:\Ashish\TestingProjects\Datasets\coffee-quality-database-from-cqi\coffeecluster.csv',\
                 )
#cols_list=['Aroma','Flavor','Aftertaste','Acidity','Body','Balance','Uniformity','Sweetness','Cupper.Points','Moisture']
    
cols_list=['Aroma','Flavor','Aftertaste','Acidity','Body','Balance','Uniformity','Sweetness','Cupper.Points','Moisture','Color']
data['Color'].fillna('None',inplace=True)
data['Color']=data['Color'].map({'Green':1,'Bluish-Green':2,'None':0,'Blue-Green':3})    


x=data[cols_list].values
y=data['Labels']

x=np.append(x,[[Aroma,Flavor,Aftertaste,Acidity,Body,Balance,Uniformity,Sweetness,CupperPoints,Moisture,Color]],axis=0)

x_std = StandardScaler().fit_transform(x)

userinput=x_std[1301,:]
x_std=np.delete(x_std,1301,axis=0)

train_x,test_x,train_y,test_y=train_test_split(x_std,y,test_size=0.2,random_state=0)

# SVM CLASSIFIER
svclassifier=SVC(kernel='rbf')
svclassifier.fit(train_x,train_y)

pred=svclassifier.predict(test_x)
print(accuracy_score(test_y,pred))

userlabel=svclassifier.predict(userinput.reshape(1,-1))[0]

userdf=data[data['Labels']==userlabel]
userdf=userdf[['Species','Owner','Country.of.Origin','Region','Farm.Name','Mill','Owner.1','Producer','Grading.Date','Variety','Certification.Body']]
