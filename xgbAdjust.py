from  sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

from sklearn.metrics import mean_squared_error
import warnings 
warnings.filterwarnings(action='ignore')
import matplotlib as mpl
mpl.rcParams["font.sans-serif"] = ["SimHei"]
import matplotlib.pyplot as plt
from xgboost import plot_importance  
import graphviz

from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from xgboost import XGBClassifier
import xgboost as xgb

import lightgbm as lgb
from sklearn.model_selection import KFold
# from matplotlib import pyplot

x_train_path="./DataSet/x_train.csv"
x_test_path="./DataSet/x_test.csv"
y_train_path="./DataSet/y_train.csv"
y_test_path="./DataSet/y_test.csv"



if __name__ == "__main__":
    x_train=pd.read_csv(x_train_path)
    x_test=pd.read_csv(x_test_path)
    y_train=pd.read_csv(y_train_path)
    y_test=pd.read_csv(y_test_path)
   
    print("\nx_train:",x_train.head())
    print("\nx_test:",x_test.head())
    print("\y_train:",y_train.head())
    print("\y_test:",y_test.head())





    
    cv_params = { 
                    'n_estimators': [100,110,120]
                }
    other_params = {'learning_rate': 0.1,  'max_depth': 4, 'min_child_weight': 6, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.05, 'reg_alpha': 0, 'reg_lambda': 1}

    # xgb_other_params = {
    #                     'metric':'rmse',
    #                     'subsample': 0.8, 'colsample_bytree':0.8,
    #                     'max_depth': 4,'num_leaves': 6, 
    #                     'reg_alpha': 0,'reg_lambda': 1,
    #                     'gamma': 0.05,'seed': 0,
    #                     'min_child_samples':16,'min_child_weight': 6,
    #                     'learning_rate': 0.02,
    #                     'n_estimators': 290,
    #                     }
    # model = xgb.XGBRegressor(**xgb_other_params)
    # model.fit(x_train, y_train)
    # y_pred = model.predict(x_test)
    # print(mean_squared_error(y_test, y_pred))  #19.7603659989499 ==>9.434325293492448

    model = lgb.LGBMRegressor(**other_params)
    optimized_xgb = GridSearchCV(estimator=model, param_grid=cv_params, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
    optimized_xgb.fit(x_train, y_train)
    evalute_result = optimized_xgb.cv_results_
    # print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_xgb.best_params_))
    print('最佳模型得分:{0}'.format(-optimized_xgb.best_score_))


    # bst = xgb.train(params, dtrain, num_round)
    # bst.save_model('./xgb.model')
    # bst2 = xgb.Booster()
    # bst2.load_model('./xgb.model')
    # # y_pred = bst.predict(dtest)
    # print(mean_squared_error(y_test, y_pred))
    

    # 7. result visualise
    # plt.figure(figsize=(12,6), facecolor='w')
    # ln_x_test = range(len(x_test))
    # plt.plot(ln_x_test, y_test, 'r-', lw=2, label=u'实际值')
    # plt.plot(ln_x_test, y_pred, 'g-', lw=4, label=u'XGBoost模型')
    # plt.xlabel(u'数据编码')
    # plt.ylabel(u'租赁价格')
    # plt.legend(loc = 'lower right')
    # plt.grid(True)
    # plt.title(u'波士顿房屋租赁数据预测')
   
    # plot_importance(model,importance_type = 'cover') 
   

    '''
    print tree
    fig,ax = plt.subplots()
    xgb.plot_tree(bst,num_trees=0)
    '''
    # xgb.plot_tree(model,num_trees=10)
    # digraph=xgb.to_graphviz(model)
    # digraph.format = 'png'
    # digraph.view('./iris_xgb')
    # plt.show()

    # dtrain = xgb.DMatrix(x_train, label=y_train)
    # dtest = xgb.DMatrix(x_test)
    # params = {'max_depth':2, 'eta':1,'objective':'reg:linear'}
    # num_round = 2






















    # model = XGBClassifier()   #使用默认参数
    # model.fit(x_train, y_train)  
    
    # y_pred = model.predict(x_test)
    # print("y_test:{}".format(y_test),type(y_test))

    # predictions = [round(value,1) for value in y_pred]
    # print("y_pred:{}".format(predictions),type(predictions),len(predictions))
    # print("y_test:{}".format(y_test.tolist()),type(y_test.tolist()),len(y_test))


    # evaluate predictions
    # accuracy = accuracy_score(y_test.tolist(), predictions)
    # print("Accuracy: %.2f%%" % (accuracy * 100.0))

