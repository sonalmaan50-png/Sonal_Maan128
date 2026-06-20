import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.svm import SVC
# Q1
df=pd.read_csv("C:\\Users\\HP\\Desktop\\agriculture_yield_dataset.csv")
print(df.shape)#How many rows and columns are present?
print(df.columns)#What are the names of all columns?
print(df.head(10))#Display the first 10 records.
#Q2
print(df.dtypes)#Check the data type of each column
print(df.isnull().values.any())#Identify whether any missing values are present
print(df.isnull().sum())#If missing values exist, mention the affected columns.
#Q3
mean_values=df.mean(numeric_only=True)
print(mean_values)
maximum_mean=mean_values.idxmax()
print(f"The highest mean value is {maximum_mean}")#Which feature has the highest mean value?
std_values=df.std(numeric_only=True)
print(std_values)
maximum_std=std_values.idxmax()
print(f"The highest standard deviation is {maximum_std}")#Which feature has the highest standard deviation?
#Q4
fig=px.histogram(df,x='rainfall_mm',marginal='box',color_discrete_sequence=['red'],title='HISTOGRAM FOR RAINFALL')
fig.update_layout(bargap=0.2)
fig.show()
# From histogram it can be concluded that the hightest value of rainfall in mm is from between 825-874.9 about 101
# Almost similar value of count from 425 to 724.9
fig=px.histogram(df,x='temperature_c',marginal='box',color_discrete_sequence=['blue'],title='HISTOGRAM FOR TEMPERATURE')
fig.update_layout(bargap=0.2)
fig.show()
# Maximum crop growth is between the temperature 21.5 to 22.4 degree celcius.
# Minimum crop growth is between the temperature range of 37.5 - 38.4.
# There is no outlier present in the data of temperature and approx average temperatur eis 27 degree celcius.
fig=px.histogram(df,x='fertilizer_kg',marginal='box',color_discrete_sequence=['green'],title='HISTOGRAM FOR FERTILIZER ')
fig.update_layout(bargap=0.2)
fig.show()
# NO outliers in the data of fertilizer_kg.
# Maximum fertilizer used in killo grams is 75-84.9.
# No uniform distribution pattern

fig=px.histogram(df,x='yield_ton_per_hectare',marginal='box',color_discrete_sequence=['pink'],title='HISTOGRAM FOR YIELD')
fig.update_layout(bargap=0.2)
fig.show()
# This graph reflects GAUSSAIN DISTRIBUTION.
# The highest frequency is between the range of 4.8 to 5.
# The highest count is 119 and 3 outliers one maximum side and other two on minimum side.

#Q5
crop_count=df['crop_type'].value_counts()
print(crop_count)
crop_counts=df['crop_type'].value_counts().reset_index()
crop_counts.columns = ["crop_type", "crop_count"]
fig=px.bar(crop_counts,x='crop_type',y='crop_count',color_discrete_sequence=['blue'],title='BAR CHART FOR CROP TYPE')
fig.update_layout(xaxis_title='CROP TYPE' ,yaxis_title='COUNT OF CROPS')
fig.show()
# The crop occured most frequently is COTTON and it appears for 311 times.

#Q6
soil_count=df['soil_type'].value_counts()
print(soil_count)
soil_counts=df['soil_type'].value_counts().reset_index()
soil_counts.columns=['Soil','Frequency']
fig=px.bar(soil_counts,x='Soil',y='Frequency',color_discrete_sequence=['pink'],title="BAR PLOT FOR SOIL TYPE")
fig.update_layout(xaxis_title='SOIL TYPE',yaxis_title='FREQUENCY')
fig.show()
#The most common soil type is CLAY with frequency of 534.

#Q7
fig=px.histogram(df,x='yield_ton_per_hectare',marginal='box',color_discrete_sequence=['magenta'],title='HISTOGRAM FOR YIELD')
fig.update_layout(bargap=0.2)
fig.show()
# YES from the histogram it can be concluded that it is a approximately normal distribution
# YES there are some noticebale outliers which are 3 - 1 at the extreme of maximum and 2 at the extreme of minimum.

#Q8
plt.title('rainfall_mm vs yield_ton_per_hectare')
sns.scatterplot(data=df, x='rainfall_mm', y='yield_ton_per_hectare', alpha=0.7, s=15);
plt.show()

plt.title('fertilizer_kg vs yield_ton_per_hectare')
sns.scatterplot(data=df, x='fertilizer_kg', y='yield_ton_per_hectare', alpha=0.7, s=15);
plt.show()

# On comparing the relationship of rainfall and fertilizer with yield we can conclude that rainfall has a stronger relationship with yield.

#Q9
le = LabelEncoder()
df['crop_type'] = le.fit_transform(df['crop_type']) 
df['soil_type'] = le.fit_transform(df['soil_type']) 
sns.heatmap(df.corr(), cmap='Reds', annot=True)
plt.title('Correlation Matrix');
plt.show()

# Three features most correlated with Crop Yield are :-
# 1) rainfall_mm (0.55)
# 2) irrigation_hours (0.54)
# 3) fertilizer_kg (0.28)

#Q10
avg_soil=df.groupby('soil_type')['yield_ton_per_hectare'].mean()
print(avg_soil)

avg_crop=df.groupby('crop_type')['yield_ton_per_hectare'].mean()
print(avg_crop)

# RICE has the highest avg yield amongst all the crop types
# LOAMY soil has the highest avg yield amongst all the soil types

#Q11
print(df.dtypes) # from the output it can be analysed that crop_type and soil_type is categorical data.

converted_data=pd.get_dummies(df,columns=['soil_type','crop_type'],dtype=int)
print(converted_data.head())

#Q12
X=converted_data.drop('yield_ton_per_hectare',axis=1)
Y=converted_data['yield_ton_per_hectare']
# Here Y - is the target variable , using yield_ton_per_hectare column as target for predicting
# X is the INPUT features

#Q13
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
print("X_train shape : ",X_train.shape)
print("X_test shape : ",X_test.shape)
print("Y_train shape : ",Y_train.shape)
print("X_test shape : ",X_test.shape)

#Q14
lr_model=LinearRegression()
lr_model.fit(X_train,Y_train)

print("Coefficients : ",lr_model.coef_)
print("Intercept : ",lr_model.intercept_)

# The highest positive coefficient is of crop type RICE