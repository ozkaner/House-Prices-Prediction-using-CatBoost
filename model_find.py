import pandas as pd
import tqdm 

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error as rmse

data = pd.read_csv("preprocessed_dataset/train.csv")
submission = pd.read_csv("preprocessed_dataset/test.csv")

X = data.drop(["SalePrice"],axis=1)
y = data["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)

models = [CatBoostRegressor(), XGBRegressor(), LinearRegression(), DecisionTreeRegressor()]
loss = [] 
for model in models:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    losses = rmse(y_test,predictions)
    loss.append(losses)

model_validation = pd.DataFrame({"Model" : ["CatBoost","XGBRegressor", "LinearRegresson", "DecisionTree"],
                                 "Losses" : loss})



