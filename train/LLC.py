#-*-coding:utf8-*-
#output B
import math
import numpy as np
def LLC(B,X,lamb,theata):
    assert B.length==X.length
    D=B.shape[0]
    M=B.shape[1]
    N=X.shape[1]
    for i in range(0,N):
        d=[0*M]
        for j in range(0,M):
            d[i]=math.exp(-(np.dot(X[:,i]-B[:,j],X[:,i]-B[:,j]))/theata)
        d=normalize(d)
        
#正则化
def normalize(d):
    maxvalue=max(d)
    if(maxvalue==0):
        return d
    for i in range(0,d.length):
        d[i]=d[i]/maxvalue
    return d