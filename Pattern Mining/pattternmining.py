import pandas as pd
from itertools import combinations
# from mlxtend.preprocessing import TransactionEncoder
# from itertools import combinations


from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction import TransactionEncoder
from sklearn.frequent_patterns import fpgrowth, association_rules

df = pd.read_csv("./data/data.csv", sep=';', dtype={'BillNo': str})
# df.head()
print("#" * 50)
print(" " * 15, "Dataset Information")
print("#" * 50)
print("The Dataset has {} columns and {} rows.".format(df.shape[1], df.shape[0]))
print("The DataFrame has {} duplicated values with {} % and {} missing values.".format(df.duplicated().sum(),round((df.duplicated().sum())/len(df)*100,2),df.isnull().sum().sum()))
print(df.info())

df.drop_duplicates(inplace=True)

df = df.rename(columns={'Itemname': 'ItemName'})
df['ItemName'] = df['ItemName'].str.lower()

#Dropping Missing rows in CustomerID column
df.dropna(subset=['CustomerID'], inplace=True)

#Dropping data with negative or zero quantity
df=df.loc[df['Quantity']>0]

#Dropping data with zero price
df=df.loc[df['Price']>'0']

#Dropping Non-product data.
df=df.loc[(df['ItemName']!='postage')&(df['ItemName']!='dotcom postage')&(df['ItemName']!='adjust bad debt')&(df['ItemName']!='manual')]


df['Date'] = pd.to_datetime(df['Date'], format="%d.%m.%Y %H:%M")
df['Price'] = df['Price'].str.replace(',','.')
df['Price'] = df['Price'].astype('float')
df['CustomerID'] = df['CustomerID'].astype('int')


df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['Day'] = df['Date'].dt.day_name()

#Creating a Total price column
df['Total price']=df.Quantity*df.Price

transaction_list = df.groupby('BillNo')['ItemName'].apply(set).tolist()
print(rules)



## Convert dataset to a suitable format for scikit-learn
te = TransactionEncoder()
te_ary = te.fit(data).transform(data)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Find frequent itemsets using FP-growth algorithm
frequent_itemsets = fpgrowth(df, min_support=0.3)

# Generate association rules
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.7)
