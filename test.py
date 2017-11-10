# -*- coding: utf-8 -*-
"""
Created on Mon Oct 02 19:37:36 2017

@author: EFF-Guo
"""
import pandas as pd
import numpy as np
import bayes

train = pd.read_excel('C:\Users\EFF-Guo\Desktop\Meal Analysis (2017).xlsx')
test = pd.read_excel('filename of test data')
#---Data cleaning of train data-------------------------------------------- 
train['Pratio']=train['P[g]']/train['P target(15%)[g]']
train['Fratio']=train['F[g]']/train['F target(25%)[g]']
train['Cratio']=train['C[g]']/train['C target(60%)[g]']

train["saltratio"]=0.0
train["saltratio"].ix[train["gender"]=="male"]=train["Salt[g]"]/8.0
train["saltratio"].ix[train["gender"]=="female"]=train["Salt[g]"]/7.0     
train=train.dropna()
train=train.drop(["gender","age","height","weight","EER[kcal]",
                  "P target(15%)[g]","F target(25%)[g]","C target(60%)[g]",
                  "E[kcal]","P[g]","F[g]","C[g]","Salt[g]"],axis=1)
#---devide train data according to class----------------------------------  
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
train_breakfast_summaries=bayes.summarizeByClass(train_breakfast_dict)
train_lunch_summaries=bayes.summarizeByClass(train_lunch_dict)
train_dinner_summaries=bayes.summarizeByClass(train_dinner_dict)
#---Data cleaning of test data------------------------------------------- 
test['Pratio']=test['P[g]']/test['P target(15%)[g]']
test['Fratio']=test['F[g]']/test['F target(25%)[g]']
test['Cratio']=test['C[g]']/test['C target(60%)[g]']

test["Saltratio"]=0.0
test["Saltratio"].ix[test["gender"]=="male"]=test["Salt[g]"]/8.0
test["Saltratio"].ix[test["gender"]=="female"]=test["Salt[g]"]/7.0 
   
test=test.drop(["gender","age","height","weight","EER[kcal]",
                  "P target(15%)[g]","F target(25%)[g]","C target(60%)[g]",
                  "E[kcal]","P[g]","F[g]","C[g]","Salt[g]"],axis=1)
#---transform dataframe to list-----------------------------------------
test_array=np.array(test)
test_list=test_array.tolist()
#---predict the class and get a list of classes------------------------
predictions = []
for i in range(len(test_list)):
    if test_list[i][0]=="breakfast":
        del test_list[i][0]
        prediction = bayes.predict(train_breakfast_dict,
                                         train_breakfast_summaries,test_list[i])
    if test_list[i][0]=="lunch":
        del test_list[i][0]
        prediction = bayes.predict(train_lunch_dict,
                                         train_lunch_summaries,test_list[i])
    if test_list[i][0]=="dinner":
        del test_list[i][0]
        prediction = bayes.predict(train_dinner_dict,
                                         train_dinner_summaries,test_list[i])
    predictions.append(prediction)
#---write the claseees into the table ------------------------------------
score = pd.DataFrame(predictions,columns=["scores"])
test = pd.read_excel('filename of test data')
test=test.reset_index(drop=True)
testscore = pd.concat([test,score],axis=1)
testscore.to_excel('filename of the test data with results')

'''calculate the accuracy using the train data
del train["Type"]
train_array=np.array(train)
train_list=train_array.tolist()
accuracy=bayes.accuracy(train_list, predictions)
print accuracy
'''