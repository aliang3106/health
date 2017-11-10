# -*- coding: utf-8 -*-
"""
Created on Thu Oct 05 18:26:35 2017

@author: EFF-Guo
"""
import pandas as pd

trainval = pd.read_excel('C:\Users\EFF-Guo\Desktop\Meal Analysis (2017).xlsx')

#---Data cleaning of train data-------------------------------------------- 
trainval['Pratio']=trainval['P[g]']/trainval['P target(15%)[g]']
trainval['Fratio']=trainval['F[g]']/trainval['F target(25%)[g]']
trainval['Cratio']=trainval['C[g]']/trainval['C target(60%)[g]']

trainval["saltratio"]=0.0
trainval["saltratio"].ix[trainval["gender"]=="male"]=trainval["Salt[g]"]/8.0
trainval["saltratio"].ix[trainval["gender"]=="female"]=trainval["Salt[g]"]/7.0     

trainval.dropna()
trainval=trainval.drop(["gender","age","height","weight","EER[kcal]",
                  "P target(15%)[g]","F target(25%)[g]","C target(60%)[g]",
                  "E[kcal]","P[g]","F[g]","C[g]","Salt[g]","Type"],axis=1)

scattercols=["Pratio","Fratio","Cratio","saltratio","number of dishes",
             "Vegetables[g]"]
'''
train_1=trainval.ix[trainval["Score(1:worst 2:bad 3:good 4:best)"]==1]
train_2=trainval.ix[trainval["Score(1:worst 2:bad 3:good 4:best)"]==2]
train_3=trainval.ix[trainval["Score(1:worst 2:bad 3:good 4:best)"]==3]
'''
axs = pd.scatter_matrix(trainval[scattercols], figsize=(12,12), c="red")
'''
#del trainval["Score(1:worst 2:bad 3:good 4:best)"]
axs = pd.scatter_matrix(trainval[:], figsize=(25,25), c="red")
'''