import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

#Read CSV
df = pd.read_csv("C:\\Users\\Dhiraj\\Desktop\\Open IIT DA\\OpenIIT - Sheet1.csv")

#Drop NaN table
df.drop(['Unnamed: 12'], axis=1, inplace=True)

#Convert Strings to number
df['Population Below 40'] = df['Population Below 40'].str.replace(",","").astype(float)
df['Population Above 40'] = df['Population Above 40'].str.replace(",","").astype(float)
df['female population'] = df['female population'].str.replace(",","").astype(float)

#Fill Null values
df.fillna(df.mean()['per capita income':], inplace=True)
#df.iloc[22,:], df.iloc[15,:] = df.iloc[15,:],df.iloc[22,:]

#Encode cities and towns
df = pd.get_dummies(df,columns=['City Type'])

#Diving datafrME
features = df.iloc[:,11:-3]
others = df.iloc[:,1:11]
last = df.iloc[:,30:]


X = df.iloc[:,1:]
Y = df.iloc[:,0]

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
features = sc_X.fit_transform(features)
others = sc_X.fit_transform(others)
last = sc_X.fit_transform(last)

population = sc_X.fit_transform(population)

#Converting Series/Arrays back to dataframe
features = pd.DataFrame(features)
others = pd.DataFrame(others)
last = pd.DataFrame(last)
population = pd.DataFrame(population)

fin2 = pd.concat([others,last], axis=1)
#Reduce dimension
#Applying PCA toh only SKUs
from sklearn.decomposition import PCA
pca = PCA(n_components=1)
pca.fit(features)
features = pca.transform(features)

pca.fit(fin2)
fin2 = pca.transform(fin2)

population = df['total population']
population = pd.DataFrame(population)

#Converting Series/Arrays back to dataframe
features = pd.DataFrame(features)
fin2 = pd.DataFrame(fin2)

others = pd.DataFrame(others)
last = pd.DataFrame(last)

#Merging data frames
fin = pd.concat([population,features], axis=1)
fin.columns = ['population', 'sales']
fin.plot.scatter(x='population',y='sales')
plt.show()

#Making Clusters
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation
from sklearn.metrics import silhouette_samples, silhouette_score
range_n_clusters = [2, 3, 4, 5, 6, 7]
for n_clusters in range_n_clusters:
    
    #clusterer = AffinityPropagation()
    #cluster_labels = clusterer.fit_predict(fin)
   
    
    clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = clusterer.fit_predict(fin)
    
    #clusterer = KMeans(n_clusters=n_clusters, random_state=0)
    #cluster_labels = clusterer.fit_predict(fin)
   
    silhouette_avg = silhouette_score(fin, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    sample_silhouette_values = silhouette_samples(fin, cluster_labels)
    

#After getting score making 3 clusters
clusterer = AgglomerativeClustering(n_clusters=4)
cluster_labels = clusterer.fit_predict(X)

