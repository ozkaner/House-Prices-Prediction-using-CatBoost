# House Prices Prediction using CatBoost

This repository contains a project for predicting house prices using the CatBoost algorithm. The data used in this project is from the Kaggle House Prices competition.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Results](#results)



## Introduction

Predicting house prices is a common regression task in machine learning. In this project, we utilize the CatBoost algorithm, a gradient boosting library that handles categorical features automatically and provides high performance with minimal hyperparameter tuning.

## Dataset

The dataset used in this project is the [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) dataset from Kaggle. It includes various features about houses and their respective sale prices.

### Files

- `train.csv`: The training set containing features and target values (sale prices).
- `test.csv`: The test set containing features only, used for making predictions.

## Installation

To run this project, you need to have Python installed along with the following libraries:

- numpy
- pandas
- catboost
- scikit-learn
- matplotlib
- optuna

You can install the required libraries using pip:


```bash
pip install numpy pandas catboost scikit-learn matplotlib optuna

```

## Model Evalution

The model's performance is evaluated using Root Mean Squared Error (RMSE) as stated in the competitions. The lower the RMSE, the better the model's performance.

## Results

he model achieved a validation RMSE of 124575.627. Detailed results and analysis can be found in the 'training.py' file.

