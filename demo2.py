from mlxtend.regressor import StackingRegressor
from mlxtend.data import boston_housing_data
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt


x, y = boston_housing_data()
x = x[:100]
y = y[:100]
# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# 初始化基模型
lr = LinearRegression()
svr_lin = SVR(kernel='linear', gamma='auto')
svr_rbf = SVR(kernel='rbf', gamma='auto')
ridge = Ridge(random_state=2019,)
models = [lr, svr_lin, svr_rbf, ridge]

params = {
    'ridge__alpha': [0.1, 1.0, 10.0],
    }
sclf = StackingRegressor(regressors=models, meta_regressor=ridge)
# 训练回归器
grid = GridSearchCV(estimator=sclf, param_grid=params, cv=5, refit=True)
grid.fit(x_train, y_train)
print(grid.best_params_, grid.best_score_)