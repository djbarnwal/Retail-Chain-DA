import pandas as pd
import numpy as np

#Read CSV
df = pd.read_csv("C:\\Users\\Dhiraj\\Desktop\\Open IIT DA\\OpenIIT - Sheet1.csv")

df.drop(['Unnamed: 12'], axis=1, inplace=True)

#Convert Strings to number
df['Population Below 40'] = df['Population Below 40'].str.replace(",","").astype(float)
df['Population Above 40'] = df['Population Above 40'].str.replace(",","").astype(float)
df['female population'] = df['female population'].str.replace(",","").astype(float)

#Fill Null values
df.fillna(df.mean()['per capita income':], inplace=True)
#df.iloc[22,:], df.iloc[15,:] = df.iloc[15,:],df.iloc[22,:]


#Encode cities
#df = pd.get_dummies(df,columns=['City Type'])
df = df[df["City Type"] == "City"]


df.drop(['City Type'], axis=1, inplace=True)

features = df.iloc[:,11:-1]
others = df.iloc[:,1:11]
last = df.iloc[:,30]

X = df.iloc[:,1:]
Y = df.iloc[:,0]

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
features = sc_X.fit_transform(features)
others = sc_X.fit_transform(others)
last = sc_X.fit_transform(last)

#Reduce dimension
from sklearn.decomposition import PCA
pca = PCA(n_components=1)
pca.fit(features)
features = pca.transform(features)


features = pd.DataFrame(features)
others = pd.DataFrame(others)
last = pd.DataFrame(last)
#X.drop([22], axis=1, inplace=True)

#Merging data frames
fin = pd.concat([others,features,last], axis=1)

corr = fin.corr()

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
    

clusterer = KMeans(n_clusters=4, random_state=10)
cluster_labels = clusterer.fit_predict(X)

 clusterer = AgglomerativeClustering(n_clusters=2)
 cluster_labels = clusterer.fit_predict(fin)

