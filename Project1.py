'''
Jessica Jorgenson
Matthew Wintersteen
CSCI 347 Project 1
Python Code for Data Analysis
'''
import numpy as np
import pandas as pd
import math
from numpy import genfromtxt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

'''
A function to compute the mean of a numerical, multidimensional data set
input as a 2-dimensional numpy array
'''
def computeMean(arr):
    mean = np.zeros(arr.shape[1])
    for a in arr:
        mean += a
    mean = mean / arr.shape[0]
    return mean

'''
A function to compute the sample covariance between two attributes that
are input as one-dimensional numpy vectors
'''
def computeCovar(v1, v2):
    v1mean = np.mean(v1)
    v2mean = np.mean(v2)
    n = np.size(v1)

    summ = 0

    for i in range(n):
        summ += (v1[i]-v1mean)*(v2[i]-v2mean)

    cov = summ/(n-1)
    
    return cov

'''
A function to compute the correlation between two attributes that are input as
one-dimensional numpy vectors
'''
def computeCorr(v1, v2):
    cov12 = computeCovar(v1,v2)
    cov1 = computeCovar(v1,v1)
    cov2 = computeCovar(v2,v2)

    corr = cov12/math.sqrt(cov1*cov2)
    return corr

'''
A function to range normalize a two-dimensional numpy array
'''
def rangeNorm(arr):
    normArr = arr
    minimum = np.amin(normArr, axis=0)
    maximum = np.amax(normArr, axis=0)
    
    normArr = normArr.astype('float32')
    
    for col in range(len(normArr)):
       for row in range(len(normArr[col,:])):
           normArr[col][row] = (normArr[col][row]-minimum[row])/(maximum[row]-minimum[row])
    return normArr

'''
A function to standard normalize a two-dimensional numpy array
'''
def standardNorm(arr):
    normArr = arr
    std = np.std(normArr, axis=0)
    mean = computeMean(normArr)

    normArr = normArr.astype('float32')
    
    for col in range(len(normArr)):
       for row in range(len(normArr[col,:])):
           normArr[col][row] = (normArr[col][row]-mean[row])/(std[row])
    return normArr

'''
A function to compute the covariance matrix of a dataset
'''
def computeCovarMatrix(arr):
    n = arr.shape[1]
    covarMatrix = np.zeros([n,n], dtype = float)
    for col in range(n):
        for row in range(n):
            covarMatrix[col][row] = computeCovar(arr[:,col],arr[:,row])
    
    return covarMatrix

'''
A function to label-encode categorical data
'''
def labelEncode(v):
    encodedV = np.zeros(len(v), dtype = float)
    stringlist = []
    for i in range(len(v)):
        d = v[i].strip()
        if (d not in stringlist):
            stringlist.append(d)
            d = len(stringlist)
        else:
            d = stringlist.index(d)
        encodedV[i] = float(d)
    if (len(stringlist) > 0):
        print("Removed Strings")
    return encodedV

#Tests all the python functions written for Part 2
def testFunc():
    a = np.array([[7,14,33,48,-1],[5,15,34,50,0],[8,17,32,41,1]])
    b = np.array(["x-large","medium","large","medium","small"])
    v1 = np.array([1,2,3,2,4,1,2,1,1])
    v2 = np.array([4,1,3,1,1,0,2,1,3])

    #The equivalent solution with libraries is commented out

    print("Testing compute mean")
    print(computeMean(a))
##    print(np.mean(a, axis=0))
    print("Testing compute covariance")
    print(computeCovar(v1,v2))
##    print(np.cov(v1,v2)[1][0])
    print("Testing compute correlation")
    print(computeCorr(v1,v2))
##    print(np.corrcoef(v1,v2))
    print("Testing range normalization")
    print(rangeNorm(a))
##    scaler = MinMaxScaler()
##    scaler.fit(a)
##    print(scaler.transform(a))
    print("Testing standard normalization")
    print(standardNorm(a))
    print("Testing compute covariance matrix")
    print(computeCovarMatrix(a))
##    print(np.cov(a.transpose()))
    print("Testing label encoding")
    print(labelEncode(b))
    

#Driver for Part 3
def main():
    print("Reading input from file")
    df = pd.read_csv('imports-85.data.csv',header=None,names=columns, na_values=['?'])
    
    #One-hot-encoding all categorical data
    df = pd.get_dummies(df, columns=categorical)
        
    for i in range(len(df.columns)):
        df.iloc[:, i].fillna(df.iloc[:, i].mean(), inplace=True)

    arr = df.to_numpy()
    colNames = list(df.columns.values)
    
    #multivariate mean
    multMean = computeMean(arr)
    f = open("q1.csv", "w")
    for a in range(len(multMean)):
        f.write("{},{}\n".format(colNames[a],multMean[a]))
    f.close()

    #covariance matrix
    matrix = computeCovarMatrix(arr)
    f = open("q2.csv", "w")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            f.write("{}".format(matrix[i][j]))
            if (j != matrix.shape[1]-1):
                f.write(",")
            else:
                f.write("\n")
    f.close()
    
    
    
columns = ['symboling','normalized-losses','make','fuel-type','aspiration','num-of-doors','body-style','drive-wheels','engine-location','wheel-base','length','width','height','curb-weight','engine-type','num-of-cylinders','engine-size','fuel-system','bore','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','price']
categorical = ['make','fuel-type','aspiration','num-of-doors','body-style','drive-wheels','engine-location','engine-type','num-of-cylinders','fuel-system']

#testFunc()
main()

