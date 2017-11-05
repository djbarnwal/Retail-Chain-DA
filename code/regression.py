import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import statsmodels.api as sm
import statsmodels.formula.api as smf
matplotlib.style.use('ggplot')

def adj_r2_score(model,y,yhat):
        """Adjusted R square — put fitted linear model, y value, estimated y value in order
        
            Example:
            In [142]: metrics.r2_score(diabetes_y_train,yhat)
            Out[142]: 0.51222621477934993
        
            In [144]: adj_r2_score(lm,diabetes_y_train,yhat)
            Out[144]: 0.50035823946984515"""
        from sklearn import metrics
        adj = 1 - float(len(y)-1)/(len(y)-len(model.coef_)-1)*(1 - metrics.r2_score(y,yhat))
        return adj

#Read CSV
df = pd.read_csv("C:\\Users\\Dhiraj\\Desktop\\Open IIT DA\\OpenIIT - Sheet1.csv")

#Drop NaN table
df.drop(['Unnamed: 12'], axis=1, inplace=True)

#Convert Strings to number
df['Population Below 40'] = df['Population Below 40'].str.replace(",","").astype(float)
df['Population Above 40'] = df['Population Above 40'].str.replace(",","").astype(float)
df['female population'] = df['female population'].str.replace(",","").astype(float)
df['per capita income'] = df['per capita income'].str.replace(",","").astype(float)

#Fill Null values
df.fillna(df.mean()['per capita income':], inplace=True)
#df.iloc[22,:], df.iloc[15,:] = df.iloc[15,:],df.iloc[22,:]

#Encode cities and towns
df = pd.get_dummies(df,columns=['City Type'])

#Diving dataframe
sales = df.iloc[:,11:-2] #Including Total Sale
others = df.iloc[:,1:11]
orginal_data = df.iloc[:,[2,8,9,31]] #Population, avg CPI, area, town/city

X = df.iloc[:,1:]
y = df.iloc[:,0:1]

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

sales = sc_X.fit_transform(sales)
orginal_data = sc_X.fit_transform(orginal_data)


#Converting Series/Arrays back to dataframe
sales = pd.DataFrame(sales)
orginal_data = pd.DataFrame(orginal_data)


sales.columns = ['Cookware', 'Crockery', 'Electronics', 'Farm Fresh', 'Fashion',
       'Food Services', 'Healthcare', 'Home Essentials', 'Home Fashion',
       'Plastics', 'Processed Food', 'Shoes', 'Sports', 'Staples',
       'Stationery', 'Toys', 'Trolley Bags', 'Utensils', 'Wellness',
       'Total Sale']

orginal_data.columns = ['total population', 'Avg. CPI for the Period', 'Area (km2)','CityorNot']

#Reduce dimension
#Applying PCA toh only SKUs
from sklearn.decomposition import PCA
pca = PCA(n_components=1)
pca.fit(sales)
sales = pca.transform(sales)

sales = pd.DataFrame(sales)
sales.columns = ['Sales']

X = orginal_data
Y = sales

'''
fin = pd.concat([orginal_data,sales], axis=1)

#Add city to fin
fin = pd.concat([fin,y], axis=1)

#Seperating some outliers
outliers = df.index.isin([15,16,17,8])
fin_without_outliers = fin[~outliers]


#Visualizing our data
from pandas.tools.plotting import parallel_coordinates

parallel_coordinates(fin,'City')
parallel_coordinates(fin_without_outliers,'City')

#Individual plots
for column in fin:
    fin.plot.scatter(x=column,y='Sales')
    
'''

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

#Predicing the Test set results
y_pred = lin_reg.predict(X_test)

from sklearn.metrics import r2_score
r2_score = r2_score(y_test, y_pred)  

adjusted_r2_score  = adj_r2_score(lin_reg,y_test,y_pred)