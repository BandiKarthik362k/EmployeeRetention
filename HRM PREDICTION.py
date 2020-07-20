#!/usr/bin/env python
# coding: utf-8

# # HUMAN RESOURCE MANAGER'S PREDICTON##

# __LIBRABRIES REQUIRED__

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# __IMPORTING THE DATAFRAME__

# In[2]:


df=pd.read_csv("C:/Users/91832/Desktop/HRP/Human_Resources.csv")
df.head()


# __DATA CLEANING__

# In[3]:


df.info()


# In[4]:


#Attrition,Over18,OverTime need to be in integers but,are in object form
#Attrition is the record of the employees left or stayed the company
#Assigning "1" to left members and "0" to stayed ones
#Simillarly converting the "Yes" and "No" objects in Over18 and OverTime to 1 and 0 respectively
df["Attrition"]=df["Attrition"].apply(lambda x:1 if x=="Yes" else 0)
df["Over18"]=df["Over18"].apply(lambda x:1 if x=="Y" else 0)
df["OverTime"]=df["OverTime"].apply(lambda x:1 if x=="Yes" else 0)


# MISSING/NAN VALUES

# In[5]:


#Checking the missing values using heatmaps
sns.heatmap(df.isnull(),cbar=False,cmap="Blues",yticklabels=False)


# SUM OF THE MISSING VALUES

# In[6]:


df.isnull().sum()


# In[7]:


df.head()


# In[8]:


df.hist(figsize=(20,20))


# In[9]:


#Using bins to get the equal width of the numeric data
df.hist(figsize=(20,20),bins=25)


# In[10]:


#From the above histogram the "EmployeeCount","StandardHours","Over18" are same for every employees
#dropping them
df.drop(["EmployeeCount","Over18","StandardHours"],axis=1,inplace=True)


# In[11]:


#Previously there are 35 columns now there are 32
df.info()


# In[12]:


#EmployeeNumber for the dataframe is not required for the prediction so dropping it
df.drop(["EmployeeNumber"],axis=1,inplace=True)


# In[13]:


df.head()


# DISTINGUSHING THE ATTRITION DATA FOR LEFT AND STAYED

# In[14]:


#taking the left data as "1" and stayed data as "0" to df_left and df_stayed respectively
df_left=df[df["Attrition"]==1]
df_stayed=df[df["Attrition"]==0]


# In[15]:


df_left.head()


# In[16]:


df_stayed.head()


# In[17]:


df.head()


# In[18]:


#from BusinessTravel we have Rarely and Frequently
#removing Travel for it to get clear outview
#Note:Not much required but part of cleaning
df["BusinessTravel"]=df["BusinessTravel"].apply(lambda x:x.replace(x,x[7:]))


# In[19]:


df.head()


# In[20]:


df.info()


# In[21]:


#Seems there are some more object types converting them to int type for modeling
#selecting the categorical values to a new dataframe naming df_cat
df_cat=df[["BusinessTravel","Department","Gender","JobRole","MaritalStatus"]]


# In[22]:


df_cat.head()


# In[23]:


#We can't use the categorical values for traing the dataset
#So converting the categorical values to numericals using OneHotEncoder


# ONEHOTENCODING

# In[24]:


from sklearn.preprocessing import OneHotEncoder
encode=OneHotEncoder()
df_cat=encode.fit_transform(df_cat).toarray()
#toarray() is used to covert into array form


# In[25]:


df_cat


# In[26]:


#Coverting the obtained array to the dataframe
df_cat=pd.DataFrame(df_cat)


# In[27]:


df_cat.head()


# In[28]:


#taking the numerical values in the df datafram to df_num
df_num=df[['Age','DailyRate','DistanceFromHome','Education','EnvironmentSatisfaction','HourlyRate','JobInvolvement','JobLevel','JobSatisfaction','MonthlyIncome','MonthlyRate','NumCompaniesWorked','OverTime','PercentSalaryHike','PerformanceRating','RelationshipSatisfaction','StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']]


# In[29]:


df_num.head()


# In[30]:


#Now we got 2 data frames for both the numerical and categorical which are converted to numerics using OneHotEncoder
#Combining the both data frames to df_final
df_final=pd.concat([df_num,df_cat],axis=1)


# In[31]:


df_final.head()


# In[32]:


#Checking for type of data
df_final.info()
#Every thing are in int and float,so we are done


# __DATA VISUALIZATION__

# CORRELATION

# In[33]:


f,ax=plt.subplots(figsize=(20,20))
sns.heatmap(df.corr(),annot=True)


# From the heatmap we have observed that the heighest correlations are 0.78,0.77,0.95,0.68
# 
# 0.78 for the correlation between "JobLevel" and "TotalWorkingYears"
# 
# 0.77 for the correlation between "MonthlyIncome" and "TotalWorkingYears"
# 
# 0.95 for the coreelation between the both "MonthlyIncome,JobLevel" and "JobLevel,MonthlyIncome"
# 
# 0.68 for the correlation between the both "Age,TotalWorkingYears"

# COUNTPLOTS TO DISTINGUISH LEFT AND STAYED EMPLOYEES BASED ON THE HIGHLY RATED CORRELATION

# In[34]:


plt.figure(figsize=[20,40])
plt.subplot(10,1,1)
#Based on Age
sns.countplot(x="Age",hue="Attrition",data=df)
plt.subplot(10,1,2)
#Based on JobLevel
sns.countplot(x="JobLevel",hue="Attrition",data=df)
plt.subplot(10,1,3)
#Based on MaritalStatus
sns.countplot(x="MaritalStatus",hue="Attrition",data=df)
plt.subplot(10,1,4)
#Based on TotalWorkingYears
sns.countplot(x="TotalWorkingYears",hue="Attrition",data=df)
plt.subplot(10,1,5)
#Based on JobInvolvement
sns.countplot(x="JobInvolvement",hue="Attrition",data=df)
plt.subplot(10,1,6)
#Based on DistanceFromHome
sns.countplot(x="DistanceFromHome",hue="Attrition",data=df)
#Based on Gender
plt.subplot(10,1,7)
sns.countplot(x="Gender",hue="Attrition",data=df)
#Based on Manager
plt.subplot(10,1,8)
sns.countplot(x="YearsWithCurrManager",hue="Attrition",data=df)
#Based on Promotions
plt.subplot(10,1,9)
sns.countplot(x="YearsSinceLastPromotion",hue="Attrition",data=df)
#Based on HourlyRate
plt.subplot(10,1,10)
sns.countplot(x="HourlyRate",hue="Attrition",data=df)


# __MODEL CREATION__

# In[35]:


df_final.head()


# MIN MAX SCALING

# In[36]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x=scaler.fit_transform(df_final)


# In[37]:


#setting the Attrition as the Target
y=df["Attrition"]


# SELECTING THE TRAINING AND THE TESTING DATA

# In[38]:


from sklearn.model_selection import train_test_split
#selecting 40% of the data for testing and remaining for training
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.10)


# MODEL USING LOGISTICREGRESSION

# In[39]:


from sklearn.linear_model import LogisticRegression as LR
A=LR(solver="liblinear").fit(x_train,y_train)


# In[40]:


yhat=A.predict(x_test)


# CONFUSION MATRIX

# In[41]:


from sklearn.metrics import confusion_matrix
B=confusion_matrix(yhat,y_test)
sns.heatmap(B,annot=True)


# ACCURACY SCORE

# In[42]:


from sklearn.metrics import accuracy_score
print("Accuracy_score using LogisticRegression =",str(accuracy_score(yhat,y_test)*100)+str("%"))


# CLASSIFICATION REPORT

# In[43]:


from sklearn.metrics import classification_report
print(classification_report(yhat,y_test))


# MODEL USING RANDOM FOREST

# In[44]:


from sklearn.ensemble import RandomForestClassifier
D=RandomForestClassifier(n_estimators=10).fit(x_train,y_train)


# In[45]:


yhat_R=D.predict(x_test)


# CONFUSION MATRIX

# In[46]:


E=confusion_matrix(yhat_R,y_test)
sns.heatmap(E,annot=True)


# ACCURACY SCORE

# In[47]:


print("Accuracy_Score using RandomForest =",str(accuracy_score(yhat_R,y_test)*100)+str("%"))


# CLASSIFICATION REPORT

# In[48]:


print(classification_report(yhat_R,y_test))


# In[ ]:





# In[ ]:




