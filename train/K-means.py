#-*-coding:utf-8-*-
'''
    CNN_Features是一个list，每个元素是一个4096维的CNN特征
    返回16384个聚簇＝4*4096，存于dict中，key是第几个簇，value是每个簇中的CNN_Features
'''
import random
import numpy

def Distance(feature,center):
    diff=feature-center
    return numpy.trace(numpy.dot(diff,diff))**0.5
def update_center(clusters,clusters_center):
    diff=0
    for key in clusters:
        new_center=clusters[key][0]
        for feature_idx in range(1,clusters[key].length):
            new_center=new_center+clusters[key][feature_idx]
        new_center=new_center/clusters[key].length
        if not(clusters_center[key]==new_center):
            diff=diff+1
        clusters_center[key]=new_center
    return clusters_center,diff
def kmeans(CNN_Features,clusters_num=16384):
    #随机选取cluster_num个作为中心
    clusters_center=random.sample(CNN_Features,clusters_num)
    clusters={}
    endSteps=1000
    iters=0
    endvalue=1
    #结束条件：迭代次数达到上限，迭代达到稳定
    while iters<endSteps and endvalue!=0:
        iters=iters+1
        for feature_i in range(0,CNN_Features.length):
            minDis=Distance(CNN_Features[feature_i],clusters_center[0])
            belongto=0
            feature_idx=0
            for center_j in range(1,clusters_num):
                dis=Distance(CNN_Features[feature_i],clusters_center[center_j])
                if(minDis>dis):
                    minDis=dis
                    belongto=center_j
                    feature_idx=feature_i
            clusters[belongto].setdefault(belongto,[]).append(CNN_Features[feature_idx])
        clusters_center,endvalue=update_center(clusters,clusters_center)
    return clusters