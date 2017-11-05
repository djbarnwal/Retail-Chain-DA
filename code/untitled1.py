import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import statsmodels.api as sm
import statsmodels.formula.api as smf
matplotlib.style.use('ggplot')

#Read CSV
df = pd.read_csv("C:\\Users\\Dhiraj\\Desktop\\Open IIT DA\\OpenIITwithGSDP.csv")

df.dropna(axis=0, how='all', inplace=True)

df = df[df['City Type'] == 'Town']
#Convert Strings to number
df['Household Income(Rs,per year)'] = df['Household Income(Rs,per year)'].str.replace(",","").astype(float)
df['per capita income'] = df['per capita income'].str.replace(",","").astype(float)


#Reset index
df.reset_index(inplace=True)
df.drop(['index'], axis=1, inplace=True)
df = df.iloc[0:-1,:]

sales = df.iloc[:,16:]

mat = pd.concat([df.iloc[:,[2,12]],sales], axis=1)

corr = mat.corr()