'''
Jessica Jorgenson
Matthew Wintersteen
CSCI 347 Project 1
Python Code for Data Analysis
'''
import numpy as np
import pandas as pd
from numpy import genfromtxt
from sklearn.impute import SimpleImputer

'''
A function to compute the mean of a numerical, multidimensional data set
input as a 2-dimensional numpy array
'''
def computeMean(arr):
    return 0

'''
A function to compute the sample covariance between two attributes that
are input as one-dimensional numpy vectors
'''
def computeCovar(v1, v2):
    return 0

'''
A function to compute the correlation between that are input as
one-dimensional numpy vectors
'''
def computeCorr(v1, v2):
    return 0

'''
A function to range normalize a two-dimensional numpy array
'''
def rangeNorm(arr):
    return normArr

'''
A function to standard normalize a two-dimensional numpy array
'''
def standardNorm(arr):
    return normArr

'''
A function to compute the covariance matrix of a dataset
'''
def computeCovarMatrix(arr):
    return covarMatrix

'''
A function to label-encode categorical data
'''
def labelEncode(arr):
    return encodedArr


def main():
    print("Reading input from file")
    df = pd.read_csv('imports-85.data.csv',header=None,names=columns, na_values=['?'])
    
    #One-hot-encoding all categorical data
    df = pd.get_dummies(df, columns=categorical)
        
    for i in range(len(df.columns)):
        df.iloc[:, i].fillna(df.iloc[:, i].mean(), inplace=True)
        
    print(df)
    
columns = ['symboling','normalized-losses','make','fuel-type','aspiration','num-of-doors','body-style','drive-wheels','engine-location','wheel-base','length','width','height','curb-weight','engine-type','num-of-cylinders','engine-size','fuel-system','bore','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','price']
categorical = ['make','fuel-type','aspiration','num-of-doors','body-style','drive-wheels','engine-location','engine-type','num-of-cylinders','fuel-system']


main()

