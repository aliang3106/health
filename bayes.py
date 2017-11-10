# -*- coding: utf-8 -*-
"""
Spyderエディタ

これは一時的なスクリプトファイルです
""" 
import numpy as np
#---devide the data according to class---------------------------------
def fea_and_class(trs):
    tr_feas1=[]
    tr_feas2=[]
    tr_feas3=[]
    tr_feas4=[]
    tr_dict={}
    for tr in trs:
        tr_fea=[float(x) for x in tr[:2]]+[float(x) for x in tr[3:]]
        tr_cl=int(tr[2])
        if tr_cl==1:
            tr_feas1.append(tr_fea)
        if tr_cl==2:
            tr_feas2.append(tr_fea)
        if tr_cl==3:
            tr_feas3.append(tr_fea)
        if tr_cl==4:
            tr_feas4.append(tr_fea)
    tr_dict[1]=tr_feas1
    tr_dict[2]=tr_feas2
    tr_dict[3]=tr_feas3
    tr_dict[4]=tr_feas4
    return tr_dict
#---calculte the mean and std of each feature according to class-------
def summarizeByClass(dataset):
    summaries = {}
    for classValue, instances in dataset.iteritems():
        summaries[classValue]= [np.mean(instances,axis=0), 
                 np.std(instances,axis=0)]
    return summaries
#---the formula of gaussian probability------------------------------------
def gaussian(x,mu,sigma):
    return (1.0/(sigma*np.sqrt(2*np.pi))*
            np.exp(-(x-mu)**2/(2*sigma**2)))
#---calculate the probability of each class-----------------------------
def classprobabilities(tr_dict,summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.iteritems():
        probabilities[classValue] =len(tr_dict[classValue])/len(tr_dict)
        for i in range(len(classSummaries[0])):
           mean = classSummaries[0][i]
           std = classSummaries[1][i]
           x = inputVector[i]
           probabilities[classValue] *= gaussian(x, mean, std)
    return probabilities
#---predict the class of vector according to the calculated probabilities---
def predict(tr_dict, summaries, inputVector):
    probabilities = classprobabilities(tr_dict, summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.iteritems():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel
#---accuracy-------------------------------------------------------------------
def accuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][2]==predictions[i]:
            correct +=1
    return (correct/float(len(testSet)))*100.0 
#---normal bayes summarize by class-------------------------------------
def normalsummarize(dataset):
    summaries={}
    for classValue, instances in dataset.iteritems():
        summaries[classValue]=[np.mean(instances,axis=0),
                               np.cov(np.array(instances).T)]
        if np.linalg.det(summaries[classValue][1])==0:
           summaries[classValue][1]=np.cov(np.array(instances).T) \
                                   +np.eye(6,6)*0.0001 
    return summaries
#---Normal bayes probabilities------------------------------------------
def normalbayes(tr_dict, summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.iteritems():
        probabilities[classValue] =                                \
           np.log(len(tr_dict[classValue])/len(tr_dict))           \
           -0.5*np.log(np.linalg.det(classSummaries[1]))           \
           -0.5*(np.mat(inputVector)-np.mat(classSummaries[0]))    \
               *np.mat(classSummaries[1]).I                        \
               *(np.mat(inputVector)-np.mat(classSummaries[0])).T
        probabilities[classValue] = probabilities[classValue][0,0]
    return probabilities 
#---predict the class of vector according to the calculated probabilities
def normalpredict(tr_dict, dataset, inputVector):
    probabilities = normalbayes(tr_dict, dataset, inputVector)
    bestLabel, bestProb = None, -1000
    for classValue, probability in probabilities.iteritems():
        if bestLabel is None or probability > bestProb:
           bestProb = probability
           bestLabel = classValue
    return bestLabel  