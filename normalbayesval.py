# -*- coding: utf-8 -*-
"""
Created on Mon Oct 02 13:39:19 2017

@author: EFF-Guo
"""
import pandas as pd
import numpy as np
import bayes
from sklearn.cross_validation import train_test_split

trainval = pd.read_excel('C:\Users\EFF-Guo\Desktop\Meal Analysis (2017).xlsx')
#---Data cleaning of train data-------------------------------------------- 
trainval['Pratio']=trainval['P[g]']/trainval['P target(15%)[g]']
trainval['Fratio']=trainval['F[g]']/trainval['F target(25%)[g]']
trainval['Cratio']=trainval['C[g]']/trainval['C target(60%)[g]']
#trainval['Eratio']=trainval['E[kcal]']/trainval['EER[kcal]']

trainval["saltratio"]=0.0
trainval["saltratio"].ix[trainval["gender"]=="male"]=trainval["Salt[g]"]/8.0
trainval["saltratio"].ix[trainval["gender"]=="female"]=trainval["Salt[g]"]/7.0     

trainval=trainval.dropna()
trainval=trainval.drop(["gender","age","height","weight","EER[kcal]",
                  "P target(15%)[g]","F target(25%)[g]","C target(60%)[g]",
                  "E[kcal]","P[g]","F[g]","C[g]","Salt[g]"],axis=1)
#---1000 kinds of validation splits---------------------------------------
accuracy=0
for iter_num in range(1000):
#---Validation Split------------------------------------------------------
    train, val = train_test_split(trainval, test_size=0.2)

    val_x = val.drop(["Score(1:worst 2:bad 3:good 4:best)"],axis=1)
    val_x_array=np.array(val_x)
    val_x_list=val_x_array.tolist()
#---devide train data according to class-------------------------------  
    train_breakfast=train.ix[train["Type"]=="breakfast"]
    train_lunch=train.ix[train["Type"]=="lunch"]
    train_dinner=train.ix[train["Type"]=="dinner"]
    del train_breakfast["Type"]
    del train_lunch["Type"]
    del train_dinner["Type"]

    train_breakfast=np.array(train_breakfast)
    train_breakfast=train_breakfast.tolist()
    train_lunch=np.array(train_lunch)
    train_lunch=train_lunch.tolist()
    train_dinner=np.array(train_dinner)
    train_dinner=train_dinner.tolist()

    train_breakfast_dict=bayes.fea_and_class(train_breakfast)
    train_lunch_dict=bayes.fea_and_class(train_lunch)
    train_dinner_dict=bayes.fea_and_class(train_dinner)
#---calculate the mean and cov according to class------------------------
    train_breakfast_summaries=bayes.normalsummarize(train_breakfast_dict)
    train_lunch_summaries=bayes.normalsummarize(train_lunch_dict)
    train_dinner_summaries=bayes.normalsummarize(train_dinner_dict)
#---predict the class and get a list of classes--------------------------    
    predictions = []
    for i in range(len(val_x_list)):
       if val_x_list[i][0]=="breakfast":
          del val_x_list[i][0]
          prediction = bayes.normalpredict(train_breakfast_dict,
                                           train_breakfast_summaries,val_x_list[i])
       if val_x_list[i][0]=="lunch":
          del val_x_list[i][0]
          prediction = bayes.normalpredict(train_lunch_dict,
                                           train_lunch_summaries,val_x_list[i])
       if val_x_list[i][0]=="dinner":
          del val_x_list[i][0]
          prediction = bayes.normalpredict(train_dinner_dict,
                                           train_dinner_summaries,val_x_list[i])
       predictions.append(prediction)
#calculate the accuracy using the validation and predictions--------------
    del val["Type"]
    val_array=np.array(val)
    val_list=val_array.tolist()
    accuracy += bayes.accuracy(val_list, predictions)
average_accuracy=accuracy/1000.0
print average_accuracy