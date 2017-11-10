# -*- coding: utf-8 -*-
"""
Created on Wed Oct 04 16:46:55 2017

@author: EFF-Guo
"""

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

trainval["saltratio"]=0.0
trainval["saltratio"].ix[trainval["gender"]=="male"]=trainval["Salt[g]"]/8.0
trainval["saltratio"].ix[trainval["gender"]=="female"]=trainval["Salt[g]"]/7.0     

trainval.dropna()
trainval=trainval.drop(["gender","age","height","weight","EER[kcal]","Type",
                  "P target(15%)[g]","F target(25%)[g]","C target(60%)[g]",
                  "E[kcal]","P[g]","F[g]","C[g]","Salt[g]"],axis=1)
accuracy=0
for iter_num in range(1000):
#---Validation Split-----------------------------------------------------------
   train, val = train_test_split(trainval, test_size=0.1)

   val_x = val.drop(["Score(1:worst 2:bad 3:good 4:best)"],axis=1)
   val_x_array=np.array(val_x)
   val_x_list=val_x_array.tolist()
#---devide train data according to class-------------------------------  
   train=np.array(train)
   train=train.tolist()

   train_dict=bayes.fea_and_class(train)
#---calculate the mean and std of each feature according to category-------
   train_summaries=bayes.summarizeByClass(train_dict)

#---predict the class and get a list of classes----------------------
   predictions = []
   for i in range(len(val_x_list)):
        prediction = bayes.predict(train_dict,train_summaries, val_x_list[i])
        predictions.append(prediction)
#calculate the accuracy using the validation and predictions---------------
   val_array=np.array(val)
   val_list=val_array.tolist()
   accuracy+=bayes.accuracy(val_list, predictions)
print accuracy/1000.0