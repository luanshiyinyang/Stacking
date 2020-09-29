from mlxtend.regressor import StackingCVRegressor
from mlxtend.data import boston_housing_data
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
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
ridge = Ridge(random_state=2019,)
lasso =Lasso()
models = [lr, svr_lin, ridge, lasso]

print("base model")
for model in models:
    score = cross_val_score(model, x_train, y_train, cv=5)
    print(score.mean(), "+/-", score.std())
sclf = StackingCVRegressor(regressors=models, meta_regressor=lasso)
# 训练回归器
print("stacking model")
score = cross_val_score(sclf, x_train, y_train, cv=5)
print(score.mean(), "+/-", score.std())

sclf.fit(x_train, y_train)
pred = sclf.predict(x_test)
print("loss is {}".format(mean_squared_error(y_test, pred)))