#-*-coding:utf-8-*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
from sklearn.svm import SVC
from numpy import array
from numpy import hstack
from numpy import array
from numpy import hstack
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Flatten
# from keras.layers.convolutional import Conv1D
# from keras.layers.convolutional import MaxPooling1D




def train_data_preprocessing(train_data):
    cleaned_data = train_data.copy(deep=True)
    # print(cleaned_data.dropna('index').shape) 11769 -> 486, 너무 많이 손실 일어남

    # print(str(cleaned_data.person.isna().sum())) -> person 이상치 11004

    # print(str(cleaned_data.new_price.isna().sum()))

    cleaned_data.drop(columns=['no',"name", "person", "auto_stick", "sil", "new_price", "date", "dom_for","method"], inplace=True)

    cleaned_data.dropna("index", inplace=True)
    cleaned_data = cleaned_data.reset_index(drop=True)

    #print(cleaned_data.shape) 이상치 제거 후 (10263,8)

    # print(len(np.unique(list(cleaned_data.name)))) 차종 갯수 확인

    # brand = list(cleaned_data.name)
    #
    # for i in range(len(brand)):
    #     brand[i] = brand[i].split(' ', 1)[0]
    # cleaned_data.name = brand
    # cleaned_data.head()

    # print(len(np.unique(list(cleaned_data.name))))브랜드 30개

    length = list(cleaned_data.length)
    temp = 0
    for i in range(len(length)):
        length[i] = length[i].replace("만km", "0000")
        length[i] = length[i].replace("천km", "000")
        length[i] = length[i].replace("km", "")
        length[i] = length[i].replace("3만ml", "48280")
        length[i] = length[i].replace("8만ml", "128747")
        length[i] = length[i].replace("10만ml", "160934")
        length[i] = length[i].replace("5만ml", "80467")
        length[i] = length[i].replace("7만ml", "112654")
        length[i] = length[i].replace("9만ml", "144840")
        length[i] = length[i].replace("6만ml", "96560")
        length[i] = length[i].replace("4만ml", "64373")
        length[i] = length[i].replace("11만ml", "177027")
        length[i] = length[i].replace("2만ml", "32186")
        length[i] = length[i].replace("13만ml", "209214")

    cleaned_data.length = length

    # print(cleaned_data.length)몇km탔나 보는거 한글단위랑 km제거하고 숫자로 치환

    cleaned_data["old_price"] = cleaned_data["old_price"].astype(float)
    cleaned_data["length"] = cleaned_data["length"].astype(float)
    cleaned_data["how_old"] = cleaned_data["how_old"].astype(float)
    cleaned_data["hp"] = cleaned_data["hp"].astype(float)
    cleaned_data["tok"] = cleaned_data["tok"].astype(float)

    # print(cleaned_data.dtypes) 수치들 전부 folat 형으로 변경

    # print(np.unique(list(cleaned_data.name)))

    # print(np.unique(list(cleaned_data.old_price))) 0같은 이상한값 있는지 없는지 확인


    cleaned_data["how_old"] = pd.Categorical(cleaned_data["how_old"])
    cleaned_data = pd.get_dummies(cleaned_data, prefix_sep='_', drop_first=True)

    #print(cleaned_data.shape) #데이터 프레임 재정의, 30개
    #print(cleaned_data.columns)



    fig, ax = plt.subplots(1,4, figsize=(16,4))
    ax[0].boxplot(list(cleaned_data.length))
    ax[0].set_title("length")

    # ax[1].boxplot(list(cleaned_data.fuel))
    # ax[1].set_title("fuel")

    # ax[1].boxplot(list(cleaned_data.how_old))
    # ax[1].set_title("how_old")

    ax[1].boxplot(list(cleaned_data.hp))
    ax[1].set_title("hp")

    ax[2].boxplot(list(cleaned_data.tok))
    ax[2].set_title("tok")

    ax[3].boxplot(list(cleaned_data.old_price))
    ax[3].set_title("old_price")


    #plt.show()#outlier를 제거하기 위해 그래프를 그려봄


    #sns.pairplot(data=cleaned_data, x_vars=["length", "hp", "tok"], y_vars="old_price", size =3) xy평면에 그리려고 했는데 잘 안되서 패스

    idx = []
    tok_ = list(cleaned_data["tok"])
    for i in range(len(tok_)):
        if(tok_[i] > 60):
            idx.append(i)

    cleaned_data = cleaned_data.drop(idx)
    cleaned_data = cleaned_data.reset_index(drop = True)

    idx = []
    price_ = list(cleaned_data["old_price"])
    for i in range(len(price_)):
        if (price_[i] > 12000):
            idx.append(i)

    cleaned_data = cleaned_data.drop(idx)
    cleaned_data = cleaned_data.reset_index(drop=True)


    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].boxplot(list(cleaned_data.length))
    ax[0].set_title("length")

    # ax[1].boxplot(list(cleaned_data.fuel))
    # ax[1].set_title("fuel")

    # ax[1].boxplot(list(cleaned_data.how_old))
    # ax[1].set_title("how_old")

    ax[1].boxplot(list(cleaned_data.hp))
    ax[1].set_title("hp")

    ax[2].boxplot(list(cleaned_data.tok))
    ax[2].set_title("tok")

    ax[3].boxplot(list(cleaned_data.old_price))
    ax[3].set_title("old_price")

    #plt.show() #outlier를 제거하기 위해 그래프를 그려봄. tok. price의 outlier 제거

    return cleaned_data


def test_data_preprocessing(train_data):
    cleaned_data = train_data.copy(deep=True)

    #print(cleaned_data.shape)
    #print(cleaned_data.dropna('index').shape) 11769 -> 486, 너무 많이 손실 일어남

    #print(str(cleaned_data.tok.isna().sum()))

    tok_mean = (float)(cleaned_data["tok"].mean())
    #print(tok_mean)

    tok = list(cleaned_data.tok)

    for i in range(len(tok)):
        tok[i] = (str)(tok[i]).replace("nan", (str)(tok_mean))

    cleaned_data.tok = tok

    #print(str(cleaned_data.tok.isna().sum()))tok 결측값을 평균으로 대체함

    hp_mean = (float)(cleaned_data["hp"].mean())
    #print(tok_mean)

    hp = list(cleaned_data.hp)

    for i in range(len(hp)):
        hp[i] = (str)(hp[i]).replace("nan", (str)(hp_mean))

    cleaned_data.hp = hp
    #print(type(cleaned_data))

    #print(str(cleaned_data.hp.isna().sum()))hp결측값을 평균으로 대체함

    #
    # print(str(cleaned_data.new_price.isna().sum()))
    #
    cleaned_data.drop(columns=['no',"name", "person", "auto_stick", "sil", "date", "dom_for","method"], inplace=True)
    cleaned_data["price"] = 0
    cleaned_data.dropna("index", inplace=True)
    cleaned_data = cleaned_data.reset_index(drop=True)
    #
    #print(cleaned_data.shape) #이상치 제거 후 (5789,6)

    #######test set은 이상치 제거하면 안된다#########

    # print(len(np.unique(list(cleaned_data.name)))) 차종 갯수 확인

    # brand = list(cleaned_data.name)
    #
    # for i in range(len(brand)):
    #     brand[i] = brand[i].split(' ', 1)[0]
    # cleaned_data.name = brand
    # cleaned_data.head()

    # print(len(np.unique(list(cleaned_data.name))))브랜드 30개

    length = list(cleaned_data.length)
    temp = 0
    for i in range(len(length)):
        length[i] = length[i].replace("만km", "0000")
        length[i] = length[i].replace("천km", "000")
        length[i] = length[i].replace("km", "")
        length[i] = length[i].replace("3만ml", "48280")
        length[i] = length[i].replace("8만ml", "128747")
        length[i] = length[i].replace("10만ml", "160934")
        length[i] = length[i].replace("5만ml", "80467")
        length[i] = length[i].replace("7만ml", "112654")
        length[i] = length[i].replace("7만ml", "112654")
        length[i] = length[i].replace("9만ml", "144840")
        length[i] = length[i].replace("6만ml", "96560")
        length[i] = length[i].replace("4만ml", "64373")
        length[i] = length[i].replace("3천ml", "4828")
        length[i] = length[i].replace("등록", "100000")

    cleaned_data.length = length

    # print(cleaned_data.length)몇km탔나 보는거 한글단위랑 km제거하고 숫자로 치환

    cleaned_data["length"] = cleaned_data["length"].astype(float)
    cleaned_data["how_old"] = cleaned_data["how_old"].astype(float)
    cleaned_data["hp"] = cleaned_data["hp"].astype(float)
    cleaned_data["tok"] = cleaned_data["tok"].astype(float)



    # print(cleaned_data.dtypes) 수치들 전부 folat 형으로 변경

    # print(np.unique(list(cleaned_data.name)))

    # print(np.unique(list(cleaned_data.old_price))) 0같은 이상한값 있는지 없는지 확인

    cleaned_data["how_old"] = pd.Categorical(cleaned_data["how_old"])
    cleaned_data = pd.get_dummies(cleaned_data, prefix_sep='_', drop_first=True)

    #print(cleaned_data.shape)# 데이터 프레임 재정의, 30개
    #print(cleaned_data.columns)

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].boxplot(list(cleaned_data.length))
    ax[0].set_title("length")

    # ax[1].boxplot(list(cleaned_data.fuel))
    # ax[1].set_title("fuel")

    # ax[1].boxplot(list(cleaned_data.how_old))
    # ax[1].set_title("how_old")

    ax[1].boxplot(list(cleaned_data.hp))
    ax[1].set_title("hp")

    ax[2].boxplot(list(cleaned_data.tok))
    ax[2].set_title("tok")

    #plt.show() #outlier를 제거하기 위해 그래프를 그려봄

    # sns.pairplot(data=cleaned_data, x_vars=["length", "hp", "tok"], y_vars="old_price", size =3) xy평면에 그리려고 했는데 잘 안되서 패스

    # idx = []
    # tok_ = list(cleaned_data["tok"])
    # for i in range(len(tok_)):
    #     if (tok_[i] > 80):
    #         idx.append(i)
    #
    # cleaned_data = cleaned_data.drop(idx)
    # cleaned_data = cleaned_data.reset_index(drop=True)


    return cleaned_data

def cost_function(X,Y,B):
    m = len(Y)
    J = np.sum((X.dot(B) - Y ** 2))/(2 * m)
    return J

def pred(x_test, newB):
    return x_test.dot(newB)

def batch_gradient_descent(X,Y,B, alpha, iterations):
    cost_history = [0] * iterations
    m = len(Y)

    for iterations in range(iterations):
        h = X.dot(B)
        loss = h-Y
        gradient = X.T.dot(loss) / m
        B = B - alpha * gradient
        cost = cost_function(X,Y,B)
        cost_history[iterations] = cost

    return B, cost_history

def r2(y_,y):
    sst = np.sum((y-y.mean())**2)
    ssr = np.sum((y_-y)**2)
    r2 = 1-(ssr/sst)
    return(r2)



if __name__ == "__main__":

    train_data_load = pd.read_csv("train.csv")
    train_data = train_data_preprocessing(train_data_load)

    test_data_load = pd.read_csv("test.csv")
    test_data = test_data_preprocessing(test_data_load)

    submission_data_load = pd.read_csv("submission.csv")




    y = train_data[["old_price"]].to_numpy()
    train_data = train_data.drop(columns=["old_price"])

    x = train_data.values
    columns = train_data.columns

    scaler = preprocessing.RobustScaler()
    tmp = scaler.fit_transform(x)
    train_data = pd.DataFrame(tmp)
    train_data.columns = columns

    x = train_data.to_numpy()


    #train 데이터 정리 완료



    x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.85, random_state=1)

    lr = LinearRegression(fit_intercept=True, normalize=True, copy_X=True)


    lr.fit(x_train, y_train)
    y_result_predict = lr.predict(x_train)

    # print(y_test)
    # print(y_result_predict)
    #
    # print("%.1lf"%(mean_squared_error(y_test , y_result_predict)))
    # rmse = np.sqrt(mean_squared_error(y_test , y_result_predict))
    # print(rmse)
    # result = 100 * (2000 - rmse) / 2000
    #
    # print("0'th model result : %.10lf"%result)





    ###### test 데이터 정리 ########

    y2 = test_data[["price"]].to_numpy()

    x2 = test_data.values
    columns2 = test_data.columns

    scaler2 = preprocessing.MinMaxScaler()
    tmp2 = scaler.fit_transform(x2)
    test_data = pd.DataFrame(tmp2)
    test_data.columns = columns2

    x2 = test_data.to_numpy()





    lr = LinearRegression(fit_intercept=True, normalize=True, copy_X=True)
    lr.fit(x,y)
    score = (lr.score(x, y))
    print("Train data's Accuracy : %.16lf"%(score))

    lr.fit(x, y)
    y_predict = lr.predict(x2)

    # for i in y_predict:
    #     print("%.2lf"%(i))



    # print("Test data's Accuracy :", format(lr.score(x_test,y_test)))
    # print("Test data's Accuracy :", format(r2_score(y_test,lr.predict(x_test))))
    #
    # print(mean_absolute_error(y_test, y_predict))
    # print(y_test)
    #
    # print("-------------------")
    # print(len(y_predict))




    # temp = list()
    # set = list()
    # for i in range(1,5790):
    #     temp.append(i)
    #     temp.append(y_predict[i-1])
    #     set.append(temp)
    #
    # result = pd.DataFrame(set,columns=['no','가격(만원)'])

    ##### GradientBoostingRegresoor로 오차 줄이기

    # num_estimators = [100,500,1000]
    # learn_rates = [0.1, 0.5, 1]
    # max_depths = [1,2,3,4]
    # min_samples_leaf = [5,10,15]
    # min_samples_split = [2,5,10]
    #
    # param_grid = {'n_estimators': num_estimators, 'learning_rate': learn_rates, 'max_depth':max_depths,\
    #               'min_samples_leaf':min_samples_leaf, 'min_samples_split':min_samples_split}
    #
    # grid_search = GridSearchCV(GradientBoostingRegressor(loss='huber'),param_grid,cv=3,return_train_score=True)
    #
    # grid_search.fit(x,y.ravel())
    #
    # print(grid_search.best_params_) #이걸로 아래 넣을 최적 파라미터를 구한다.

    gb = GradientBoostingRegressor(min_samples_leaf=3, min_samples_split=5, learning_rate=0.1, max_depth=5,\
                                   n_estimators=10)
    gb.fit(x_train,y_train)
    gb_acc = gb.score(x_test, y_test)

    print("Train data's Accuracy1 :", gb_acc)

    gb.fit(x,y)
    y_gb_predict = gb.predict(x2)



    #
    # for i in y_gb_predict:
    #      print("%.0lf"%(i))


    ####### RandomForestRegressor로 오차 줄이기
    # param = dict()
    # param["max"] = 0
    # param["n_estimator"] = 0
    # param["random_state"] = 0


    rfr = RandomForestRegressor(min_samples_leaf=3, min_samples_split=5, max_depth=3,\
                                   n_estimators=10)
    rfr.fit(x_train,y_train)
    acc = rfr.score(x_test, y_test)
    print("Train data's Accuracy2 :",acc)





    # if acc > param["max"]:
        #     param["max"] = acc
        #     param["n_estimator"] = i
        #     param["random_state"] = j

    #print(param)
    #print("Train data's Accuracy2 :", format(rfr.score(x, y)))

    rfr.fit(x, y)
    y_predict2 = rfr.predict(x2)

    for i in y_predict2:
         print("%.2lf"%(i))

    rfr.fit(x_train,y_train)
    y_predict2_test = rfr.predict(x_test)

    regressor = SVR(kernel='rbf',)




    rmse = mean_squared_error(y_test , y_predict2_test)**0.5
    print(rmse)
    result = 100 * (2000 - rmse) / 2000
    #
    print("0'th model result : %.10lf"%result)







    ########## 다중회귀 경사하강법 ##########

    # sc = StandardScaler()
    # X = sc.fit_transform(x)
    #
    # m = 7000
    # f = 3
    # X_train = X[:m, :f]
    # X_train = np.c_[np.ones(len(X_train), dtype='int64'),X_train]
    # y_train = y[:m]
    # X_test = X[m:, :f]
    # X_test = np.c_[np.ones(len(X_test), dtype='int64'),X_test]
    # y_test = y[m:]
    #
    # B = np.zeros(X_train.shape[1])
    # alpha = 0.005
    # iter_ = 2000
    # newB, cost_history = batch_gradient_descent(X_train, y_train, B, alpha, iter_)
    #
    # y_ = pred(X_test,newB)
    #
    # r2_sc = r2(y_, y_test)
    # print("R2 score",r2_sc)
    #
    # ans_ = pred(X_test[3], newB)
    #
    # print(ans_)

    ############### CNN을 써보자 ###############

    dataset = list()
    x = list(x)

    for i in range(0, len(x)):
        temp_ = list()
        temp = list(x[i])
        y_ = y[i][0]
        y_ = (float)(y_)
        temp_ = temp.append(y_)


        dataset.append(temp)


    # # define model
    # model = Sequential()
    # model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Flatten())
    # model.add(Dense(50, activation='relu'))
    # model.add(Dense(1))
    # model.compile(optimizer='adam', loss='mse')
    #
    #
    #
    #














