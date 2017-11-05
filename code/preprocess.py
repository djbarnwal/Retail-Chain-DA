import pandas as pd

df3 = df = pd.read_excel("C:\\Users\\Dhiraj\\Desktop\\Open IIT DA\IIT DATA ANALYTICS Competition (data set).xlsx","Detailed data")
df2 = pd.read_excel("C:\\Users\\Dhiraj\\Desktop\\Open IIT DA\IIT DATA ANALYTICS Competition (data set).xlsx","Additional Data")

city = df["City Name"].unique()
city = pd.Series(city)
city = city.dropna()

for index, row in df.iterrows():
    if(not pd.isnull(df.at[index, 'City Name'])):
        curr  = df.at[index, 'City Name']
    else:
        df.at[index, 'City Name'] = curr
             

df = df.iloc[:452,:]
df= df.pivot(index='City Name', columns='Category of SKU',values='Sale')
df = df.dropna(axis=1, how='all')
df = df.reset_index()
df.rename(columns={'City Name': 'City'}, inplace=True)

data = pd.merge(df, df2, on='City', how='outer')

T_sale = df3["Total City Sale"].unique()
T_sale = pd.Series(T_sale)
T_sale = T_sale.dropna()
T_sale.index = range(len(T_sale))
data["Total Sale"] = T_sale
 
corr = data.corr()
    

'''
writer = pd.ExcelWriter('C:\\Users\\Dhiraj\\Desktop\\Open IIT DA\openiitdata.xlsx')
data.to_excel(writer,'Sheet1')
writer.save()
'''
