#!/usr/bin/env python
# coding: utf-8

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import pandas as pd
import seaborn as sns
%matplotlib inline  

#load csv
heart_df = pd.read_csv("combined.csv")
heart_df

# drop columns:
    # added index row (from converting pands df to csv)
    # ssn (documentation: replaced by dummy value of 0)
    # dummy (same value as trestbps)
    # restckm (documentation: irrelevant)
    # exerckm (documentation: irrelevant)
    # thalsev (documentation: not used)
    # thalpul (documentation: not used)
    # earlobe (documentation: not used)
    # lvx1, lvx2, lvx3, lvx4, lvf, cathef (documentation: not used)
    # junk
    # name (documentation: last name of patient replaced by dummy string "name")
        # note that dummy equivalent to trestbps: https://www.sciencedirect.com/science/article/pii/S2352914820300125

heart_df = heart_df.drop(["Unnamed: 0", "ssn", "dummy", "restckm", "exerckm", "thalsev", "thalpul", "earlobe", "lvx1", "lvx2", "lvx3", "lvx4", "lvf", "cathef", "junk", "name"], axis=1)
heart_df

# create a copy of df for making changes (preserve original)
heart_df_copy = heart_df.copy()
heart_df_copy["num"].value_counts()


# **Note: num = diagnosis of heart disease (angiographic disease status)**
# * Originally 0, 1, 2, 3, 4 with 0 indicating no heart disease (< 50% artery (?) diameter narrowing) and 1, 2, 3, 4 indicating various levels of artery diameter narrowing (> 50% diameter narrowing)
# * Combine 1, 2, 3, 4 into the value 1 (i.e. clip col at 1) --> **update: 0 = no heart disease, 1 = heart disease**
# 
# **Source for rationale to combining varying levels of (artery?) diameter narrowing that indicate diff. levels of heart disease (but all indicate presence): https://www.sciencedirect.com/science/article/pii/S2352914820300125**

heart_df_copy.num.clip(0, 1, inplace=True)
heart_df_copy.num.value_counts()


#recreating pncaden col --> documentation: the sum of cols 5-7 (in this case painloc, painexer, relrest)
heart_df_copy["pncaden"] = heart_df_copy[["painloc", "painexer", "relrest"]].apply(lambda x: x.painloc + x.painexer + x.relrest, axis=1)
heart_df_copy

#CATEGORICAL CONVERSION (SKIP OVER)
# list of vars that should be categorical (but when read in assumed by pandas to be numeric with dtype = float64)
# categorical_var_list = ["sex", "painloc", "painexer", "relrest", "cp", "htn", "smoke", "fbs", "dm", "famhist", "restecg", "dig", "prop", "nitr", "pro", "diuretic",
                        # "proto", "exang", "xhypo", "slope", "thal", "num", "lmt", "ladprox", "laddist", "diag", "cxmain", "ramus", "om1", "om2", "rcaprox", "rcadist"]

# mass_change_dtype(heart_df_copy, categorical_var_list, "category")
# create a dict with each of the categorical cols as keys and value for each of the key-value pairs "category" in order to pass this dict to df.astype() to convert dtypes

# pandas categorical data type: https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html
# categorical_var_dict = {}
# for cat_var in categorical_var_list:
#     categorical_var_dict[cat_var] = "category"

# update the data type for the categorical vars
# heart_df_copy = heart_df_copy.astype(categorical_var_dict)
# heart_df_copy.info(verbose=True)

#FUNCTION to check of missing values
def check_na(df):
    '''
    Purpose: identify missing values in a dataframe and place the column index, column name, and the number of missing values for each col into a tuple and append it to a list
            * can identify which columns with lots of missing values
            
    INPUT
    df: dataframe
    
    OUTPUT:
    col_sums: list of tuples --> tuple for each column with column index, column name, number of missing values
    '''
    
    # create a boolean df from df arg showing values that are Na (True if Na, False if a value)
    bool_df = df.isna()
    
    # created empty list col_sums
    col_sums = []
    
    # for each of the columns and their index in the original df -> sum cols of corresponding bool df (True coerced to 1) and append to list col_sums
    for ix, col in enumerate(df.columns):
        col_sum = bool_df[col].sum()
        
        # df.columns[ix] to index into list of col names returned by df.columns and return the name of that column
        # source: https://stackoverflow.com/questions/43068412/retrieve-name-of-column-from-its-index-in-pandas
        col_sums.append((ix, df.columns[ix], col_sum))
    
    # returns list of col index, name, and number of missing values
    return col_sums

def remove_missing(df, threshold):
    """
    PURPOSE: remove columns from a dataframe with at least a given number of missing values
    
    INPUT:
    df: dataframe
    threshold: lower bound (exclusive) for the acceptable number of missing values
    
    OUTPUT:
    df: dataframe with the columns with at least a given number of missing values (given the threshold) removed 
    """
    
    # create a list that contains tuples (one for each column -> (column index, column name, number of missing values))
    # calls the check_na() function
    missing_vals = check_na(df)
    # print(missing_vals, type(missing_vals))
    
    # for each item (tuple) in the list missing_values:
        # if the number of missing values (the third elt – index 2 – in each tuple) is at least a given threshold:
            # drop that column from the dataframe (accessed using the second elt – index 1 – in each tuple)
    for item in missing_vals:
        if item[2] >= threshold:
            df.drop(item[1], axis=1, inplace = True)
    
    return df

heart_df_copy = remove_missing(heart_df_copy, 540)
heart_df_copy

# drop more irrelevant columns
heart_df_copy = heart_df_copy.drop(["id", "ekgmo", "ekgday", "ekgyr", "slope", "cmo", "cday", "cyr", "lmt", "ladprox", "laddist", "cxmain", "om1", "rcaprox", "rcadist"], axis=1)
heart_df_copy
#most cleaned df, 899 rows x 37 columns 

corr_main = heart_df_copy.corr()
sns.heatmap(heart_df_copy.corr())

#LOGISTIC REGRESSION with num (heart disease present) and thalach (max heart rate)

heart_df_copy_log_regression = heart_df_copy[['num', 'thalach']]
#drop NaNs
heart_df_copy_log_regression = heart_df_copy_log_regression.dropna()
heart_df_copy_log_regression = heart_df_copy_log_regression.reset_index(drop = True)
heart_df_copy_log_regression

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression

X = heart_df_copy_log_regression.drop(['num'], axis = 1) #Only thalach
#thalach is independent var
Y = heart_df_copy_log_regression['num'].astype(str) #Only num
#num is target var 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.20, random_state=0)
#random_state = 0 will select records randomly
#data broken into two parts ratio 80:20, 80% data used for model training and 20% used for testing model

#model development and prediction 
#initialize model
#train logistic model with training data 
log_reg = LogisticRegression(class_weight='balanced' , random_state=0).fit(X_train, Y_train)
#perform prediction on test thalach data using regression model/training data
y_pred=log_reg.predict(X_test) #produces predicted num 
y_pred

#confusion matrix
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_pred, Y_test) #predicted num and actual num for test patients 
#in matrix 
cnf_matrix
#The dimension of this matrix is 2*2 because this model is binary classification. 
#You have two classes 0 and 1. Diagonal values represent accurate predictions, 
#while non-diagonal elements are inaccurate predictions. 
#In the output, 52 and 58 are actual predictions, and 33 and 26 are incorrect predictions.
#false positive is type 1 error, outcome where model incorrectly predicts pos class when it is actually neg
#false negative is type 2 error, outcome where model incorrectly predicts neg class when it is actually positive 

#confusion matrix plot
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
#in heat map, 52 is TN, 33 is FN, 26 is FP, 58 is TP 

#evaluation metrics 
#use results of matrix to determine the accuracy, precision and recall of the model
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
#out of 844 patient records used, model correctly predicted whether someone had heart disease 65%
print("Precision:", metrics.precision_score(Y_test, y_pred, pos_label='1'))
#model correctly predicts if patients will suffer from heart disease based on max heart rate 69% of the time
print("Recall:", metrics.recall_score(Y_test, y_pred, pos_label="1"))
#model can identify patients with heart disease in test set 63.7% of the time

#ROC curve, plot of true positive rate against false positive rate, shows tradeoff between sensitivity and specificity 
#AUC score 1 represents perfect classifier, and 0.5 represents a worthless classifier.

y_pred_proba = log_reg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(Y_test,  y_pred_proba, pos_label='1')
auc = metrics.roc_auc_score(Y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
#auc value of 0.74, model is a good classifier: able to determine more true positives and true negatives than false positives/false negatives

#logistic regression steps/code referenced from https://www.datacamp.com/community/tutorials/understanding-logistic-regression-python

