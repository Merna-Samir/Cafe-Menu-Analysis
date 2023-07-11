#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt 
import warnings as wn
from sklearn.preprocessing import MinMaxScaler
from mlxtend.frequent_patterns import apriori, association_rules
from pymining import seqmining
from prefixspan import PrefixSpan
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from scipy.special import inv_boxcox
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler #To convert data to array

import warnings as wn
wn.filterwarnings("ignore")
sns.set_style("darkgrid")


# In[2]:


DateInfo = pd.read_csv(r'DateInfo.csv')
DateInfo_df = pd.DataFrame(DateInfo)


# In[3]:


# Return first 5 elements of data
DateInfo_df.head()


# In[4]:


# Return last 5 elements of data
print(DateInfo_df.tail())


# In[5]:


# info of our data
# Print the name columns of df & Print ncol and nrow
print(DateInfo_df.info(),"\n")


# In[6]:


#Making sure the data is clean
#Checking the possibility of data containing empty cells
cols = DateInfo_df.columns
DateInfo_df[cols].isnull().sum()


# In[7]:


DateInfo_df.duplicated()


# In[8]:


DateInfo_df.CALENDAR_DATE = pd.to_datetime(DateInfo_df.CALENDAR_DATE)
DateInfo_df


# ### Data DateInfo clean

# In[9]:


Transaction = pd.read_csv(r'Cafe - Transaction - Store.csv')
Transaction_df = pd.DataFrame(Transaction)
Transaction_df


# In[10]:


# Return first 5 elements of data
Transaction_df.head()


# In[11]:


# info of our data
# Print the name columns of df & Print ncol and nrow
print(Transaction_df.info(),"\n")


# In[12]:


#Making sure the data is clean
#Checking the possibility of data containing empty cells
cols = Transaction_df.columns
Transaction_df[cols].isnull().sum()


# In[13]:


Transaction_df.duplicated()


# In[14]:


Transaction_df.CALENDAR_DATE = pd.to_datetime(Transaction_df.CALENDAR_DATE)
Transaction_df


# ### Data Transaction clean

# In[15]:


SellMetaData = pd.read_csv(r'Cafe - Sell Meta Data.csv')
SellMetaData_df = pd.DataFrame(SellMetaData)
SellMetaData_df


# In[16]:


# Return first 5 elements of data
SellMetaData_df.head()


# In[17]:


# Return last 5 elements of data
print(SellMetaData_df.tail())


# In[18]:


# info of our data
# Print the name columns of df & Print ncol and nrow
print(SellMetaData_df.info(),"\n")


# In[19]:


#Making sure the data is clean
#Checking the possibility of data containing empty cells
cols = SellMetaData_df.columns
SellMetaData_df[cols].isnull().sum()


# In[20]:


SellMetaData_df.duplicated()


# ### Data SellMetaData clean

# # Merge the datasets

# In[21]:


# Merge the datasets on the common column "key"
merged_data = pd.merge(DateInfo_df, Transaction_df , on="CALENDAR_DATE")
cafe_data = pd.merge(merged_data ,SellMetaData_df , on=["SELL_ID",'SELL_CATEGORY' ])

CafeData_df = pd.DataFrame(cafe_data)
CafeData_df


# In[22]:


# Return first 5 elements of data
CafeData_df.head()


# In[23]:


# Return last 5 elements of data
print(CafeData_df.tail())


# In[24]:


# Discription of our data
CafeData_df.describe()


# In[25]:


fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111)

corr_matrix = CafeData_df.corr()

sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")

plt.show()


#  notes:
# from the correlation matrix we know the correlation(positive or negative)and non crrelation between the data The correlation 
# 
# 1.- There is a linear correlation among them.
# 
# 2.- There are nulls in the feature holiday.
# 
# 3.-between (AVERAGE_TEMPERATURE,YEAR)is positive crrelation=0.08 The correlation 
# 
# 4.-between (SELL_ID,PRICE)is negative crrelation=-0.76 
# 
# 5.-Between (YEAR,IS_WEEKEND)the correlation coefficient is very close to zero, indicating that there is almost no correlation between the two variables
# 
# 6.- QTY is highly linear correlated to sell_id, sell_cat 
# 
# 7.- week negative crrelation is_weekend and price ,and so on..
# 

# # Cluster

# In[27]:


# Drop irrelevant variables
droped_data = CafeData_df.drop(['STORE','SELL_ID','ITEM_ID','ITEM_NAME','CALENDAR_DATE'], axis=1)

# Convert categorical variables to numerical 
data = pd.get_dummies(droped_data)

# Standardize the numerical variables
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
data_scaled


# In[28]:


#Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=20)
clusters = dbscan.fit_predict(data_scaled)

#Analyze the resulting clusters
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
print(f"Number of clusters: {n_clusters}")
print(f"Cluster sizes: {pd.Series(clusters).value_counts()}") 


# # Association Rule

# In[29]:


# Select columns to use for association rule mining 
#only the catigoral data
columns_for_ar = ["HOLIDAY", "IS_WEEKEND", "IS_SCHOOLBREAK", "IS_OUTDOOR", "STORE", "ITEM_NAME"]
# Create one-hot encoded DataFrame
#get_dummies() function converts each categorical column in data[columns_for_ar] into a set of binary (0 or 1) columns
onehot_data = pd.get_dummies(CafeData_df[columns_for_ar])

# Generate frequent itemsets
frequent_itemsets = apriori(onehot_data, min_support=0.01, use_colnames=True)
#we use the association_rules() function from the mlxtend library to generate association rules from the frequent itemsets.
# metric="lift" use to measures how much more likely the consequent of a rule is given the antecedent
#if ilft=1 means that the antecedent and consequent are independent, while a value greater than 1 means that they are positively correlated
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Print the results
print(rules.head())


# # Sequential Pattern

# In[30]:


# Select the HOLIDAY, SELL_CATEGORY,PRICE and ITEM_NAME columns as the sequence to mine
sequence = CafeData_df[["HOLIDAY","SELL_CATEGORY","PRICE","ITEM_NAME"]].apply(tuple, axis=1).tolist()
# Perform sequential pattern discovery using prefixspan
ps = PrefixSpan(sequence)
#the code is likely using it to identify patterns of data that occur frequently in a dataset. The number "2" passed as an argument to the "frequent
#looking for patterns that occur at least two times in the dataset
patterns = ps.frequent(2)

# Print the frequent patterns
df_patterns= pd.DataFrame(patterns)
df_patterns


# # percentage change in quantity demanded and price

# In[31]:


# Create a linear regression model
X = CafeData_df['PRICE']
X = sm.add_constant(X)
y = CafeData_df['QUANTITY']
model = sm.OLS(y, X).fit()

# Calculate the price elasticity
price_coeff = model.params['PRICE']
avg_price = np.mean(CafeData_df['PRICE'])
avg_quantity = np.mean(CafeData_df['QUANTITY'])
price_elasticity = price_coeff * (avg_price / avg_quantity)

# Print the results
print("Price elasticity:", price_elasticity)


# If the price elasticity of demand is greater than 1, meaning that a change in price will have a relatively large effect on the quantity demanded. If the price elasticity of demand is less than 1, meaning that a change in price will have a relatively small effect on the quantity demanded.
# 
# form the run we know that Price elasticity: 2.185646352692894 which mean that there is large effect on the quantity demanded

# # Visualization work and extraction of results from it

# In[32]:


fig,axis=plt.subplots(nrows=1,ncols=3 , figsize=(16,5))
ax = sns.violinplot(data = CafeData_df,x='SELL_ID',y='PRICE' ,ax=axis[0])
plt.title('Price by sell id')
ax =sns.violinplot(data = CafeData_df,x='SELL_ID',y='QUANTITY',ax=axis[1])
plt.title('Qulitty by sell id')
ax =sns.scatterplot(data=CafeData_df,x='PRICE',y='QUANTITY',hue="SELL_ID",s=150,alpha=0.75 ,ax=axis[2])
plt.title("Bubble Chart",size=30)


# Of this 3 charts we can observe that:
# 
# 1.- The 1070 has a different set of price and quantity (more expensive and more quantity) than the combos.
# 
# 2.- 1070 has different price distribution but the others are similar.
# 
# 3.- The 1070 quantity is more spread. All products have different qty distribution.
# 
# 4.- Split by sell_id would be a good idea to optimize the prices since they have different distributions

# # Transformations

# In[33]:


def split_and_boxcox(dataset,feature,SELL_ID):
    dataset = dataset[dataset['SELL_ID'] == SELL_ID].copy()
    params=[]
    for x in feature:
        feature_box= x+"_boxcox"
        dataset[feature_box],params_x = stats.boxcox(dataset[x]) 
        params.append(params_x)
    return dataset,params


# In[34]:


df = CafeData_df


# In[35]:


#display the out layer
fig,axis=plt.subplots(nrows=1,ncols=3 , figsize=(10,5))
ax=df.boxplot('QUANTITY',ax=axis[0])
ax=df.boxplot('PRICE',ax=axis[1])
ax=df.boxplot('AVERAGE_TEMPERATURE',ax=axis[2])
plt.show()


# 
# I used the box cox transformation to give a more symmetrical shape to both features.
# 
# Now I will use the z-score to scale down the numerical features
# 

# In[36]:



df['HOLIDAY'].fillna('None', inplace=True)  # Replace missing values with 'None'
df = pd.get_dummies(df, columns=['HOLIDAY'])

scaler_price = StandardScaler()
scaler_qty = StandardScaler()
scaler_avg = StandardScaler()
#Product 1070 (I will optimize the product 1070, for the rest I would follow the same procedure)
df_1070,param_1070 = split_and_boxcox(df,['QUANTITY','PRICE','AVERAGE_TEMPERATURE'],1070)
df_1070['QTY_T'] = scaler_qty.fit_transform(df_1070[['QUANTITY']])
df_1070['PRICE_T'] = scaler_price.fit_transform(df_1070[['PRICE']])
df_1070['AVG_TEMPERATURE_T'] = scaler_avg.fit_transform(df_1070[['AVERAGE_TEMPERATURE']])
df_1070


# In[37]:


#display the out layer
fig, axis = plt.subplots(ncols=2, nrows=1, figsize=(10,5))
ax = df_1070.boxplot('QTY_T', ax=axis[0])
ax = df_1070.boxplot('PRICE_T', ax=axis[1])


# 
# 
#  After the transformations the outliers had dissapear.
# 

# In[38]:


# Label encode year
encoder = LabelEncoder()
df_1070['YEAR']  = encoder.fit_transform(df_1070['YEAR'] )
df_1070


# # Model (Linear regression analysis )

# In[39]:


X = df_1070[['PRICE_T', 'HOLIDAY_New Year', 'IS_WEEKEND', 'IS_SCHOOLBREAK', 'IS_OUTDOOR', 'YEAR', 'AVG_TEMPERATURE_T']]
y = df_1070['QTY_T']
X = sm.add_constant(X)  # Add constant term
model = sm.OLS(y, X)

# Fit model and print summary
results = model.fit()
print(results.summary())

# Create partial regression plot
fig = plt.figure(figsize=(12, 8))
fig = sm.graphics.plot_partregress_grid(results, fig=fig)

# Create regression plot for price vs quantity
fig = plt.figure(figsize=(12, 8))
fig = sm.graphics.plot_regress_exog(results, 'PRICE_T', fig=fig)


# results:
# 
#  is_schoolbreak and avg_temperature have a p-value > 0.1 so they don´t affect to the good fit of the model
# 
# 

# ### The next step is to remove those features that are not relevant for the model

# In[40]:



X1 = df_1070[['PRICE_T', 'HOLIDAY_New Year', 'IS_WEEKEND', 'IS_OUTDOOR']]
y1 = df_1070['QTY_T']

X1 = sm.add_constant(X1)  # Add constant term
model1 = sm.OLS(y1, X1)

# Fit model and print summary
results1 = model1.fit()
print(results1.summary())

# Create partial regression plot
fig = plt.figure(figsize=(12, 8))
fig = sm.graphics.plot_partregress_grid(results1, fig=fig)

# Create regression plot for price vs quantity
fig = plt.figure(figsize=(12, 8))
fig = sm.graphics.plot_regress_exog(results1, 'PRICE_T', fig=fig)


# # Model (Logistic Regression )

# In[41]:


Log_df= CafeData_df

encoder=LabelEncoder()
Log_df['ITEM_NAME']=encoder.fit_transform(Log_df['ITEM_NAME'])
Log_df['HOLIDAY'].fillna('None', inplace=True)  # Replace missing values with 'None'
Log_df = pd.get_dummies(CafeData_df, columns=['HOLIDAY'])


# In[42]:


# Prepare the data
X2 = Log_df[["IS_WEEKEND", "IS_SCHOOLBREAK","IS_OUTDOOR","ITEM_NAME"]]
y2= Log_df["SELL_CATEGORY"]
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
print(X2_train.shape, X2_test.shape)
# Train the model
model = LogisticRegression()
model.fit(X2_train, y2_train)

# Make predictions
y2_pred = model.predict(X2_test)

# Evaluate the model
accuracy = accuracy_score(y2_test, y2_pred)
print("Accuracy:", accuracy)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x2 = np.linspace(-10, 10, 100)
y2 = sigmoid(x2)

plt.plot(x2, y2)
plt.show()


# In[43]:


# Visualize the results
cm = confusion_matrix(y2_test, y2_pred)
sns.heatmap(cm, annot=True, cmap="Blues")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Confusion Matrix")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




