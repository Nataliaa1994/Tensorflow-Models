# -*- coding: utf-8 -*-
"""
Created on Sun May  2 18:31:21 2021

@author: nan
"""

import sys
import scipy
import plotly
import plotly.express as px
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import statsmodels.formula.api as sm
import plotly.graph_objects as go
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten

'''The goal of this project is to build a model based on 11 predictors (predictive attributes), 
that allows us to determine the value an outcome variable. 
The attribute whose outcome value we will predict is "stroke", which tells us whether the
patient has had a stroke or not.

To calculate the value of "Stroke", we will use 11 explanatory variables:
    1) id: unique identifier
    2) gender: "Male", "Female" or "Other"
    3) age: age of the patient
    4) hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
    5) heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
    6) ever_married: "No" or "Yes"
    7) work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
    8) Residence_type: "Rural" or "Urban"
    9) avg_glucose_level: average glucose level in blood
    10) bmi: body mass index
    11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
'''

#reading data
filePath = "C:/Users/nan/Documents/Python Scripts/healthcare-dataset-stroke-data.csv"

raw_data = pd.read_csv(filePath, sep=",",encoding = "ISO-8859-1")

print(raw_data.head())

#Once the set is loaded, check its size and preview each observation to check if everything loaded correctly.:
raw_data.shape
print(str(raw_data.shape[0]) + ' rows.')
print(str(raw_data.shape[1]) + ' columns.')

   


#Checking the types of each variable.:
raw_data.dtypes

#This summary will be our starting point for further analysis.:
raw_data.describe().T
print(raw_data.describe().T)
raw_data.info()

# Drop the id column
raw_data.drop(columns = ["id"], inplace = True)
raw_data.head()
print(raw_data.head())

#Let's now focus on finding the missing values in the set. 
#Let's check which variables are empty and how many empty variables there are.:
nulls_summary = pd.DataFrame(raw_data.isnull().any(), columns=['Nulls'])   
nulls_summary['Num_of_nulls [qty]'] = pd.DataFrame(raw_data.isnull().sum())   
nulls_summary['Num_of_nulls [%]'] = round((raw_data.isnull().mean()*100),2)   

print(nulls_summary)

print(str(round(raw_data.isnull().any(axis=1).sum()/raw_data.shape[0]*100,2))+'% observations contain missing values in the data..')

''' The only variable with 'Nan' values ​​is 'bmi' - 201 missing values, 
which is approximately 3.93% of all values ​​in this column.
Missing data can be handled as follows:
1)Delete rows with missing data.
2)Replacing missing values with estimated data.
    
As mentioned earlier, ignoring rows with missing data can lead to 
inconsistent results, since the deleted data may be critical for 
further calculations and may contain important observations.
'''
#We replace the value of 'Nan'
#with the median of the other column values present in the dataset.

median = raw_data['bmi'].median()
print(median)
raw_data['bmi'].fillna(median, inplace=True)

#50% have a body mass index less than or equal to 28.1 and 
#the remaining 50% have a whole body mass index equal to or greater than 28.1.
print(raw_data)

raw_data.isnull().sum()

#So now no 'Nan' values present in the data

raw_data.info()





#Single variable analysis:

# 1) Analysis of numerical variables:

'''At this point, we will examine numerical variables. 
We will focus on analysing the charts to identify possible anomalies and detect outliers. 
Additionally, we will examine the distribution of the variables using the Scipy library.'''

#The "age" variable:
# Histogram
plt.figure(figsize=(13,7))
sns.set(font_scale=1.4, style="whitegrid")
sns.distplot(raw_data['age'], kde = False, bins = 30, color = '#eb6c6a').set(title = 'Histogram - "age"', xlabel = 'age', ylabel = 'number of observations')
plt.show()

# Density graph
plt.figure(figsize=(13,7))
sns.set(font_scale=1.4, style="whitegrid")
sns.kdeplot(raw_data['age'], shade = True, color = '#eb6c6a').set(title = 'Density graph - "age"', xlabel = 'atak', ylabel = '')
plt.show()

# Box plot
plt.figure(figsize=(13,7))
sns.set(font_scale=1.4, style="whitegrid")
sns.boxplot(raw_data['age'], color = '#eb6c6a').set(title = 'Box plot - "age"', xlabel = 'age')
plt.show()

# Test for normality of distribution
# Assumed significance level alpha = 0.05.
if(scipy.stats.normaltest(raw_data['age'])[1] < 0.05):
    print('The variable does not come from the normal distribution, therefore use an alternative assumption.')
else:
    print('The variable fits the normal distribution therefore use this assumption')
    


#The "bmi" variable:
# Histogram
plt.figure(figsize=(13,7))
sns.set(font_scale=1.4, style="whitegrid")
sns.distplot(raw_data['bmi'], kde = False, bins = 30, color = '#eb6c6a').set(title = 'Histogram - "bmi"', xlabel = 'bmi', ylabel = 'number of observations')
plt.show()

# Density graph
plt.figure(figsize=(13,7))
sns.set(font_scale=1.4, style="whitegrid")
sns.kdeplot(raw_data['bmi'], shade = True, color = '#eb6c6a').set(title = 'Density graph - "bmi"', xlabel = 'bmi', ylabel = '')
plt.show()

# Box plot
plt.figure(figsize=(13,7))
sns.set(font_scale=1.4, style="whitegrid")
sns.boxplot(raw_data['bmi'], color = '#eb6c6a').set(title = 'Box plot - "bmi"', xlabel = 'bmi')
plt.show()

# Test for normality of distribution
# Assumed significance level alpha = 0.05.
if(scipy.stats.normaltest(raw_data['bmi'])[1] < 0.05):
    print('The variable does not come from the normal distribution, therefore use an alternative assumption')
else:
    print('The variable fits the normal distribution therefore use this assumption')

#The "avg_glucose_level" variable:
# Histogram
plt.figure(figsize=(13,7))
sns.set(font_scale=1.4, style="whitegrid")
sns.distplot(raw_data['avg_glucose_level'], kde = False, bins = 30, color = '#eb6c6a').set(title = 'Histogram - "avg_glucose_level"', xlabel = 'avg_glucose_level', ylabel = 'number of observations')
plt.show()

# Density graph
plt.figure(figsize=(13,7))
sns.set(font_scale=1.4, style="whitegrid")
sns.kdeplot(raw_data['avg_glucose_level'], shade = True, color = '#eb6c6a').set(title = 'Density graph - "avg_glucose_level"', xlabel = 'avg_glucose_level', ylabel = '')
plt.show()

# Box plot
plt.figure(figsize=(13,7))
sns.set(font_scale=1.4, style="whitegrid")
sns.boxplot(raw_data['avg_glucose_level'], color = '#eb6c6a').set(title = 'Box plot - "avg_glucose_level"', xlabel = 'avg_glucose_level')
plt.show()

# Test for normality of distribution
# Assumed significance level alpha = 0.05.
if(scipy.stats.normaltest(raw_data['avg_glucose_level'])[1] < 0.05):
    print('The variable does not come from the normal distribution, therefore use an alternative assumption.')
else:
    print('The variable fits the normal distribution therefore use this assumption.')
    
    
    
#Categorical variables:
    
#The „gender” variable

print('Distribution of the "gender" variable:')
print(raw_data['gender'].value_counts(normalize = True))

plt.figure(figsize=(10,7))
sns.set(font_scale=1.4)
sns.countplot(raw_data['gender'], palette = 'Blues_d', order = raw_data['gender'].value_counts().index).set(title = 'Density graph - "gender"', xlabel = 'gender', ylabel = 'number of observations')
plt.show()

# The data consists of more women's data, although the gender difference is not huge.

#The „work_type” variable

print('Distribution of the "work_type" variable:')
print(raw_data['work_type'].value_counts(normalize = True))

plt.figure(figsize=(10,7))
sns.set(font_scale=1.4, style="whitegrid")
sns.countplot(raw_data['work_type'], palette = ['#eb6c6a', '#f0918f', '#f2a3a2', '#f5b5b4', '#f7c8c7']).set(title = 'Density graph - "work_type"', xlabel = 'work_type', ylabel = 'number of observations')
plt.show()

# We can see a large number of people work in the private sector.

#The „smoking_status” variable

print('Distribution of the "smoking_status" variable:')
print(raw_data['smoking_status'].value_counts(normalize = True))

plt.figure(figsize=(10,7))
sns.set(font_scale=1.4, style="whitegrid")
sns.countplot(raw_data['smoking_status'], palette = ['#eb6c6a', '#f0918f', '#f2a3a2', '#f5b5b4', '#f7c8c7']).set(title = 'Density graph - "smoking_status"', xlabel = 'smoking_status', ylabel = 'number of observations')
plt.show()

# WOW , nice to see that most of the people quoted don't smoke

#The „Residence_type” variable

print('Distribution of the "Residence_type" variable:')
print(raw_data['Residence_type'].value_counts(normalize = True))

plt.figure(figsize=(10,7))
sns.set(font_scale=1.4, style="whitegrid")
sns.countplot(raw_data['Residence_type'], palette = "deep").set(title = 'Density graph - "Residence_type"', xlabel = 'Residence_type', ylabel = 'number of observations')
plt.show()

# The minimal difference between people living in the city and rural.


#The „ever_married” variable

print('Distribution of the "ever_married" variable:')
print(raw_data['ever_married'].value_counts(normalize = True))

plt.figure(figsize=(10,7))
sns.set(font_scale=1.4,style="whitegrid")
sns.countplot(raw_data['ever_married'], palette="deep").set(title = 'Density graph - "ever_married"', xlabel = 'ever_married', ylabel = 'number of observations')
plt.show()

'''"#8000ff","#da8829"'''
# Analysis of dependencies between variables:
    
#1) Correlation analysis between numerical variables:
    
corr_num = pd.DataFrame(scipy.stats.spearmanr(raw_data.select_dtypes(include = ['float', 'int']))[0],
                        columns = raw_data.select_dtypes(include = ['float', 'int']).columns,
                        index = raw_data.select_dtypes(include = ['float', 'int']).columns)

plt.figure(figsize=(15,6))
sns.set(font_scale=1)
sns.heatmap(corr_num.abs(), cmap="Reds", linewidths=.5).set(title="Heatmap of Spearman's rank correlation coefficient")
plt.show()

# 2)Analysis of the relationship between categorical variables

def CramersV(tab):
    a = scipy.stats.chi2_contingency(tab)[0]/sum(tab.sum())
    b = min(tab.shape[0]-1, tab.shape[1]-1,)
    return(np.sqrt(a/b))

def CalculateCrammersV(tab):
    ret = []
    for m in tab:
        row = []
        for n in tab:
            cross_tab = pd.crosstab(tab[m].values,tab[n].values)
            row.append(CramersV(cross_tab))
        ret.append(row)
    return pd.DataFrame(ret, columns=tab.columns, index=tab.columns)
crammer = CalculateCrammersV(raw_data[['gender', 'ever_married', 'work_type','Residence_type','smoking_status']])

plt.figure(figsize=(15,6))
sns.set(font_scale=1.4)
sns.heatmap(crammer, cmap="Reds", linewidths=.5).set(title='Heatmap of the Crammer dependency coefficient')
plt.show()



'''Now we will remove the value "Other" from the variable "Gender", 
because we are only interested in the concrete variable "Gender".
'''

raw_data = raw_data[raw_data["gender"].str.contains("Other")==False]

# Mapping of categorical variables
'''The next step is to change the values of the categorical variables to have numerical values.
'''
raw_data['gender'] = raw_data['gender'].map({'Male':0, 'Female':1})
raw_data['Residence_type'] = raw_data['Residence_type'].map({'Urban':0, 'Rural':1})
raw_data['smoking_status'] = raw_data['smoking_status'].map({'formerly smoked':2, 'never smoked':0, 'smokes':1, 'Unknown':3})
raw_data['ever_married'] = raw_data['ever_married'].map({'Yes':1, 'No':0})
raw_data['work_type'] = raw_data['work_type'].map({'Private':0, 'Self-employed': 1, 'Govt_job':2, 'children':3, 'Never_worked':4})
raw_data.info()

#plot heat map
plt.figure(figsize=(14,10))
sns.heatmap(raw_data.corr(method='pearson'), annot=True)
plt.show() 

'''From the map we can see negative correlation between age and work_type,
  also work_type and bmi. Stroke and age has a positive correlation similarly. Many other variables
  have such correlation values so we cannot remove any variables.
'''


'''Let see which gender is more at risk to having a stroke. The boxplot will be help for us:
'''

sns.catplot(x='stroke', y="age", hue = 'gender', kind="box", data=raw_data)
'''We can see that older females are more at risk to having a stroke.
'''


## Now we want built a model so we split the data into X and Y planes:

X = raw_data.drop(['stroke'], axis=1)
Y = raw_data.stroke

X.head()
Y.head()
print(f'X shape: {X.shape}')
print(f'Y shape: {Y.shape}')



#Split the dataset into training and test datasets.
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

#Build a the base model
baseline_model = Sequential()
baseline_model.add(Dense(10, activation='relu', input_shape=(X_train.shape[1],)))
baseline_model.add(Dense(7, activation='relu'))
baseline_model.add(Dense(1, activation='sigmoid'))

'''Above I used the simplest dense layer, combining all the units of the previous layer with all of the next.'''
baseline_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
baseline_history = baseline_model.fit(X_train, Y_train, epochs=100, batch_size=30, verbose=0)


#Evaluate the accuracy of the model based on the test dataset.
loss, acc = baseline_model.evaluate(X_test, Y_test, verbose=0)
print('Test accuracy: %.3f' % acc)

''' I think that is a good test accuracy. Next, let us discuss K-fold Cross
 Validation with TensorFlow and Keras'''

from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold

# Define the K-fold Cross Validator

kfold = KFold(n_splits=10, shuffle=True)
acc_per_fold =[]
loss_per_fold=[]
#K-fold Cross Validation model evaluation
fold_no = 1
print(f'X shape: {X.shape}')
for train, test in kfold.split(X, Y):
    
  train=[x for x in train.tolist() if x in X.index.values.tolist()]
  test=[x for x in test.tolist() if x in X.index.values.tolist()]
  
  # Define the model architecture
  model = Sequential()
  model.add(Dense(10, activation='relu', input_shape=(X_train.shape[1],)))
  model.add(Dense(7, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))


  # Compile the model
  model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {fold_no} ...')
  
  # Fit data to model
  history = model.fit(X.loc[train], Y.loc[train],
              batch_size=30,
              epochs=15,
              verbose=1)
  '''We next replace the “test loss” print with one related to what we’re doing. 
  Also, we increase the fold_no:'''
  # Generate generalization metrics
  scores = model.evaluate(X.loc[train], Y.loc[train], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_per_fold.append(scores[1] * 100)
  loss_per_fold.append(scores[0])

  # Increase fold number
  fold_no = fold_no + 1
# == Provide average scores ==
  print('------------------------------------------------------------------------')
  print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')
'''In our case, the model produces accuracies of 90-100%.'''

