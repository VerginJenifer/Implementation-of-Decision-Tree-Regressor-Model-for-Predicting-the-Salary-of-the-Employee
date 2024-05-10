# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Load the data using pd.read_csv('salary_dataset.csv').
2. Check the dataset's first few rows and info.
3. Preprocess categorical variables, like "Position", using Label Encoding or One-Hot Encoding.
4. Define features (X) and the target variable (y), such as "Position" and "Level" for features and "Salary" for the target.
5. Split the data into training and testing sets.
6. Make predictions on the testing data.
7. Evaluate the model's performance using metrics like MSE and R-squared.
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: D Vergin Jenifer
RegisterNumber: 212223240174
import pandas as pd
data=pd.read_csv("/content/Salary.csv")
data
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
*/
```

## Output:
![326874055-15f1e3e2-f49e-4ff3-b3f8-c3d5409bd869](https://github.com/VerginJenifer/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/136251012/877af0f8-e206-4748-a09e-df26ddea4251)
![326874150-97b8f8ec-0acc-42f5-80ff-a8d861b6ff48](https://github.com/VerginJenifer/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/136251012/9e6e651c-0f8d-4d24-a1f3-f04c1b55fa30)
![326874271-3b191cf6-ff9f-4185-99ae-029aa3eb12fc](https://github.com/VerginJenifer/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/136251012/0195fa50-9ab3-493e-b05d-1c439b8bd75b)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
