# CoffeeData-Clustering

<h4><b>Objective:</h4></b> To cluster coffee farms in clusters according to the similarity of features of their coffee produce.

<h4><b>Dataset:</h4></b> https://www.kaggle.com/volpatto/coffee-quality-database-from-cqi

<h4><b>Description:</b></h4>
The idea behind this project is to enable coffee manafacturing companies,for example: Nescafe,Starbucks etc. or any such company, to discover different farms which provide the same quality of coffee as their usual coffee raw material provider. The basic idea is to tackle a few problems that can arise in situations such as: 
  <br>
	a) Suppose a coffee manafacturing company with a <b>manafacturing plant in India</b> imports its coffee beans from suppose <b>Farm A     			which is located in Columbia</b>. This burdens the company with a huge transportation cost. The clustering program will show the 				 company what are the other options availa<nble to it to import a similar quality of coffee beans. Lets assume a <b> Farm B located in Sri Lanka </b> produces a similar type of coffee, then it will be grouped with the Farm A.<br> 
		 <b>Basically the company can shift its import from a farm far away to a farm which is closer geographically and cut down on                transportation cost.</b><br><br>
	b) <b>This may allow the company to procure coffee beans at a cheaper rate.</b><br><br>
	c) <b>Procuring coffee beans from different countries will help reduce the monopoly in the coffee suppliers market.</b><br><br>
	The program not only clusters the data but also classifies the user-inputted data (features of coffee) into the clusters formed by the  clustering algorithm.<br><br>
<h4><b>Technology:</b></h4><br>
Variable Reduction - PCA<br>
Data Visualization - t-SNE , Matplotlib, Seaborn<br>
Clustering Algorithm - Agglomerative Clustering <br>
<br>
I tested out different clustering algorithms with different number of components and found Agglomerative Clustering to provide comparatively satisfactory results.
<br>Other algos tested include: K-Means, Affinity Propagation, DBSCAN.
<br>
Classification algorithm - Support Vector Classifier

	

	
