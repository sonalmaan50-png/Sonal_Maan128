import pandas as pd
import numpy as np 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
#Part A: Dataset Understanding
#Q1)load and display first five records
df = pd.read_csv("C:\\Users\\HP\\Desktop\\Dataset 2.csv")
print(df.head())
#Q2)number of rows and colums in dataset
print('Shape:',df.shape)
#Q3)column names
print('\nColumns:')
print(df.columns)
#Q4)numerical and categorical features
print("Numerical columns:")
print(df.select_dtypes(include=['int64', 'float64']).columns)

print("\nCategorical columns:")
print(df.select_dtypes(include=['object']).columns)
#Q5)dataset contains missing values
print(df.isnull().sum())
print("Total missing values:", df.isnull().sum().sum())
#Part B: Exploratory Data Analysis
#Q6)average age of users
print("Average Age:", round(df['Age'].mean(),2))
#Q7)average watch hours per week
print("Average watch hours:", round(df['WatchHoursPerWeek'].mean(),2))
#Q8)average monthly spending of users
print("Average Monthly Spending:", round(df['MonthlySpend'].mean(),2))
#Q9)number of users in each subscription category
print("Number of users:",df['SubscriptionType'].value_counts())
#Q10)percentage of users who renewed their subscriptions.
renewed = (df['SubscriptionRenewed'] == "Yes").sum()
percentage = (renewed / len(df)) * 100
print(f"Renewal Percentage:",percentage)
#Part C: Data Preparation
#Q11)categorical features into numerical form
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['SubscriptionType'] = le.fit_transform(df['SubscriptionType'])
df['FavoriteGenre'] = le.fit_transform(df['FavoriteGenre'])
df['SubscriptionRenewed'] = le.fit_transform(df['SubscriptionRenewed'])
print(df.head())
#Q12)Define the feature set (X) and target variable (y) for subscription renewal prediction.
X = df.drop(['UserID', 'SubscriptionRenewed', 'MonthlySpend'], axis=1)
y = df['SubscriptionRenewed']

print(X.head())
print(y.head())

#Q13)Split the dataset into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Training size:",len(X_train))
print("Testing size:",len(X_test))
#Q14)Train a Decision Tree model to predict whether a user will renew their subscription
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
print("Decision tree trained successfully")
#Part D: Decision Tree Classification
#Q15)Evaluate the model using accuracy
pred = dt.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print(f"accuracy=",round(accuracy,2))
#Q16)Generate and interpret the confusion matrix
conf_matrix = confusion_matrix(y_test, pred)
print(f"Confusion matrix=",conf_matrix)
#Interpretation:
#49 correctly predicted no renewal
#34 correctly predicted renewal
#33 false positives
#34 false negatives
#Part E: K-Nearest Neighbors (KNN)
#Q17)Train a KNN classifier with K = 5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
print("KNN trained successfully")
#Q18)Compare the accuracy of KNN with the Decision Tree model.
knn_acc = accuracy_score(y_test, knn_pred)
print("Decision Tree:", round(accuracy,2))
print("KNN:", round(knn_acc,2))
if accuracy > knn_acc:
    print("Decision Tree performed better")
else:
    print("KNN performed better")
#Part F: Linear Regression
#Q19)Train a Linear Regression model to predict monthly spending.
X_reg = df.drop(['UserID', 'MonthlySpend'], axis=1)
y_reg = df['MonthlySpend']
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)
lr = LinearRegression()
lr.fit(X_train_reg, y_train_reg)
print("Lineaar Regression trained successfully")
#Q20)Predict the monthly spending for a new user and interpret the result.
import pandas as pd
new_user = pd.DataFrame([{
    'Age': 25,
    'Gender': 0,
    'SubscriptionType': 1,
    'WatchHoursPerWeek': 20,
    'DevicesUsed': 4,
    'FavoriteGenre': 0,
    'AdClicks': 18,
    'SubscriptionRenewed': 1
}])
predicted_spend = lr.predict(new_user)
print("Predicted Monthly Spending:", predicted_spend)

# BUSINESS REFLECTION QUESTIONS

# 1. Which factors appear to influence subscription renewal the most?
# Factors such as Subscription Type, Watch Hours Per Week,Monthly Spending, Age, and Devices Used may have a strong influence on whether a user renews their subscription.

# 2. Why is subscription renewal a classification problem?
# Subscription renewal is a classification problem because the target variable has discrete categories: Yes or No.
# The model predicts which category a user belongs to.

# 3. Why is monthly spending a regression problem?
# Monthly spending is a regression problem because it is a continuous numerical value. The model predicts an amount rather than a category.

# 4. Which algorithm performed better for renewal prediction?
# Compare the accuracy scores of Decision Tree and KNN.
# The algorithm with the higher accuracy performed better.
# (Replace this comment with your actual result after running the models.)
# Example:
# Decision Tree Accuracy = 0.82
# KNN Accuracy = 0.78
# Therefore, Decision Tree performed better.

# 5. How could the platform use these predictions to improve customer retention?
# Netflix can identify users who are likely not to renew their subscriptions and target them with personalized recommendations,discounts, special offers, or engagement campaigns to improve retention.