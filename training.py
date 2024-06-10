import pandas as pd
import tqdm 
import numpy as np 

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error 

import optuna 

data = pd.read_csv("preprocessed_dataset/train.csv")
submission = pd.read_csv("preprocessed_dataset/test.csv")

X = data.drop(["SalePrice"],axis=1)
y = data["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)


def objective(trial):

    param = {
        "iterations": trial.suggest_int("iterations", 100, 600),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 1e-1),
        "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-3, 1),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "random_strength": trial.suggest_loguniform("random_strength", 1e-3, 10),
        "bagging_temperature": trial.suggest_loguniform("bagging_temperature", 0.01, 10.0),
        "od_type": trial.suggest_categorical("od_type", ["IncToDec", "Iter"]),
    }

    model = CatBoostRegressor(**param, verbose=0)
    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50, verbose=0)
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    return rmse

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100, show_progress_bar=True)

print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")


best_params = trial.params

model_optimizerd = CatBoostRegressor(**best_params, verbose=100)
model_optimizerd.fit(X_train, y_train)


y_pred = model_optimizerd.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
print(f"Root Mean Squared Error: {rmse}")