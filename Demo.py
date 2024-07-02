import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def Load_Data(Path):
    Data = pd.read_csv(Path)
    print("Size of data set is : ",Data.shape)
    return Data

def Analysis(Data):
    X = Data[['TV', 'Radio', 'Newspaper']]
    Y = Data['Sales']

    return X, Y

def Training(X_train, Y_train):
    model = LinearRegression()
    model.fit(X_train, Y_train)
    return model

def Evaluatemodel(model, X_test, Y_test):

    Y_Prediction = model.predict(X_test)
    r2 = r2_score(Y_test, Y_Prediction)
    return r2

def main():
    print("-----------------Created by Mahesh Pawar---------------")

    Path = "Advertising.csv"
    Data = pd.read_csv(Path)
    print("Size of data set is : ",Data.shape)

    X, Y = Analysis(Data)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)

    model = Training(X_train, Y_train)
    r2 = Evaluatemodel(model, X_test, Y_test)

    print("Goodness of fit using R2 method is : ", r2)

    print("Slope of Regression line is :",model.coef_)
    print("Y Intercept of Regression line is :",model.intercept_)

if __name__ == "__main__":
    main()