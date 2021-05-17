from  sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings 
warnings.filterwarnings(action='ignore')
import matplotlib as mpl
mpl.rcParams["font.sans-serif"] = ["SimHei"]
import matplotlib.pyplot as plt
# from xgboost import plot_importance  
# import graphviz

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

# from sklearn.model_selection import GridSearchCV
# from xgboost import XGBRegressor
# import xgboost as xgb

# import lightgbm as lgb
# from sklearn.model_selection import KFold
# # from matplotlib import pyplot

# from catboost import CatBoostRegressor
# import catboost as cb

from sklearn.ensemble import RandomForestRegressor
# from sklearn.externals import joblib
import joblib
def model_lr(x_train,y_train):
    lr = LinearRegression()
    lr.fit(x_train,y_train)
    return lr

def model_gbdt(x_train,y_train):
    gbdt_other_params = {
                        'loss':'ls',
                        'subsample':0.6,
                        'max_depth': 9,
                        'min_samples_split':2,
                        'min_samples_leaf':2,
                        'learning_rate': 0.03,
                        'n_estimators': 500
                        }
       
    gbdt = GradientBoostingRegressor(**gbdt_other_params)
    gbdt.fit(x_train,y_train)
    return gbdt

def model_xgb(x_train,y_train):
    xgb_other_params = {
                       
                        'subsample': 0.8, 'colsample_bytree':0.8,
                        'max_depth': 4,
                        'reg_alpha': 0,'reg_lambda': 1,
                        'gamma': 0.05,'seed': 0,
                        'min_child_weight': 6,
                        'learning_rate': 0.02,
                        'n_estimators': 290 
                      }
    xgb_model= xgb.XGBRegressor(**xgb_other_params)
    xgb_model.fit(x_train, y_train)
    return xgb_model


def model_lgb(x_train,y_train):
    lgb_other_params = {'boosting_type':'gbdt',
                        'metric':'rmse',
                        'subsample': 0.95, 'colsample_bytree':1,
                        'bagging_freq': 1,
                        'max_depth': 5,'num_leaves': 11, 
                        'min_data_in_leaf': 20,
                        'reg_alpha': 0.01,'reg_lambda': 0.6,
                        'min_child_samples':16,'min_child_weight': 0.001,
                        'learning_rate': 0.02,
                        'n_estimators': 2800
                       }
    gbm = lgb.LGBMRegressor(**lgb_other_params)
    gbm.fit(x_train, y_train)

    return gbm

def model_cab(x_train,y_train):
    cab_other_params = {
                        'loss_function':'RMSE','eval_metric':'RMSE',
                        'max_depth': 5, 
                        'reg_lambda': 0.6,
                        'learning_rate': 0.05,
                        'n_estimators': 1500
                       }

    cab =cb.CatBoostRegressor(**cab_other_params)
    cab.fit(x_train, y_train)
    return cab

def Weighted_method(test_pre1,test_pre2,test_pre3,test_pre4,w):

    Weighted_result = w['y_pred_cab']*pd.Series(test_pre1)+w['y_pred_lgb']*pd.Series(test_pre2)+w['y_pred_xgb']*pd.Series(test_pre3)+w['y_pred_gbdt']*pd.Series(test_pre4)
    print(Weighted_result )
    return Weighted_result

FilePath="./Data/FactorCarSix.csv"

if __name__ == "__main__":
    
    pd.read_csv(FilePath)
    factor_data=pd.read_csv(FilePath)

    # print(factor_data)

    # boston = datasets.load_boston()
    # data = pd.DataFrame(boston['data'])
    # data.columns = boston['feature_names']
    # data['price']= boston['target']             
    y=factor_data.pop('score');factor_data.pop('id');x=factor_data
    # y=pow(factor_data.pop('score'),6);factor_data.pop('id');x=factor_data
    # # print(data)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    # cab=model_cab(x_train,y_train)
    # joblib.dump(cab, './Predict/FiveModel/cab_model.pkl') 

    # cab.save_model('./Predict/model/catboost_model.dump')



    # y_pred_cab_train= cab.predict(x_train)
    # y_pred_cab_test= cab.predict(x_test)

    # # print("cab_mae:",mean_absolute_error(y_test, y_pred))  #mae:mae 2.147458888468061


    # lgb=model_lgb(x_train,y_train)
    # joblib.dump(lgb, './Predict/FiveModel/lgb_model.pkl') 


    # y_pred_lgb_train= lgb.predict(x_train)
    # y_pred_lgb_test = lgb.predict(x_test)

    # # print("lgb_mae:",mean_absolute_error(y_test, y_pred))  #mae:2.400218192759189

    # xgb=model_xgb(x_train,y_train)
    # joblib.dump(xgb, './Predict/FiveModel/xgb_model.pkl') 



    # y_pred_xgb_train= xgb.predict(x_train)
    # y_pred_xgb_test= xgb.predict(x_test)
    # # print("lgb_mae:",mean_absolute_error(y_test, y_pred))  #mae:2.3201818709279975

    gbdt=model_gbdt(x_train,y_train)
    joblib.dump(gbdt, './model/lidar_v6.pkl') 
    y_pred_gbdt_train = gbdt.predict(x_train)  #404
    y_pred_gbdt_test = gbdt.predict(x_test)    #102

    print("gbdt_mae:",mean_absolute_error(y_test, y_pred_gbdt_test))  #mae:2.0927250698796556

    # # lr=model_lr(x_train,y_train)
    # # y_pred = lr.predict(x_test)
    # # print("gbdt_mae:",mean_absolute_error(y_test, y_pred))  #3.7507121808389092
    
    # # w = [0.25,0.25,0.25,0.25]
    

    # pred_test=pd.DataFrame({"y_pred_cab":y_pred_cab_test,"y_pred_lgb":y_pred_lgb_test,"y_pred_xgb":y_pred_xgb_test,"y_pred_gbdt":y_pred_gbdt_test})
    # # print(pred_test)
    # weight_pred=pred_test.apply(lambda x:x/x.sum() ,axis=1)  #总和标准化
    # # print(weight_pred)
    # weight_pred= Weighted_method(y_pred_cab_test,y_pred_lgb_test,y_pred_xgb_test,y_pred_gbdt_test,weight_pred)   #result add
    # print("weight_pred_mae:",mean_absolute_error(y_test,weight_pred))  # 2.1170302254120834

    # #stack_fusion
    # stack_x_train = pd.DataFrame()
    # stack_x_train['Method_1'] = y_pred_cab_train
    # stack_x_train['Method_2'] = y_pred_lgb_train
    # stack_x_train['Method_3'] = y_pred_xgb_train   
    # stack_x_train['Method_4'] = y_pred_gbdt_train  #train

    # stack_x_test = pd.DataFrame()
    # stack_x_test['Method_1'] = y_pred_cab_test
    # stack_x_test['Method_2'] = y_pred_lgb_test
    # stack_x_test['Method_3'] = y_pred_xgb_test   
    # stack_x_test['Method_4'] = y_pred_gbdt_test  #train

    # print(stack_x_train.shape)
    # print(stack_x_test.shape)
    # print(stack_x_test)

    # stack_lr=model_lr(stack_x_train,y_train)
    # joblib.dump(stack_lr, './Predict/FiveModel/stack_lr.pkl') 


    
    # stack_pred=stack_lr.predict(stack_x_test)
    # print("stack_mae:",mean_absolute_error(y_test, stack_pred))  #mae:2.1501818709279975

    # 7. result visualise
    plt.figure(figsize=(12,6), facecolor='w')
    ln_x_test = range(len(x_test))
    plt.figure(1)
    plt.plot(ln_x_test[:500], y_test[:500], 'r-', lw=2, label=u'实际值')
    # plt.plot(ln_x_test[:500], weight_pred[:500], 'g-', lw=4, label=u'weight模型')
    plt.plot(ln_x_test[:500], y_pred_gbdt_test[:500], 'b-', lw=4, label=u'stack模型')

    # plt.scatter(ln_x_test[:500], y_test[:500], 'r-', lw=2, label=u'实际值')
    # # plt.plot(ln_x_test[:500], weight_pred[:500], 'g-', lw=4, label=u'weight模型')
    # plt.scatter(ln_x_test[:500], stack_pred[:500], 'b-', lw=4, label=u'stack模型')
    # plt.figure(2)
    # plt.scatter(ln_x_test[:500], y_test[:500],  label=u'实际值')
    # # plt.plot(ln_x_test[:500], weight_pred[:500], 'g-', lw=4, label=u'weight模型')
    # plt.scatter(ln_x_test[:500], stack_pred[:500], label=u'stack模型')


    plt.xlabel(u'数据编码')
    plt.ylabel(u'置信度')
    plt.legend(loc = 'lower right')
    plt.grid(True)
    plt.title(u'算法结果预测')
    plt.show()

    


























