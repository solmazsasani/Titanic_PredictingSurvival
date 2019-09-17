'''

GOAL OF THIS PROJECT IS TO PREDICT IF A PASSENGER SURVIVED THE SINKING
OF TITANIC OR NOT. FOR EACH DATA SET, A VALUE OF 0 OR 1 IS PREDICTED
FOR THIS VARIABLE.

'''

#required python libraries are imported
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#First training data is read
train = pd.read_csv('train.csv')
#print(train.columns)

#First 5 rows are checked to get an idea of variables
#print(train.head())

#FIRST-> DATA IS CHECKED FOR ANY MISSING VALUES:
#sns.heatmap(train.isnull(),yticklabels = False,cbar = False,cmap = 'viridis')

#SECOND-> DATA IS EXPLORED TO GET A BETTER SENSE OF DATA:
#sns.set_style('whitegrid')
#sns.countplot(x='Survived',hue='Sex',data = train, palette='RdBu_r')
#sns.countplot(x='Survived',hue='Pclass',data = train, palette = 'rainbow')
#sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)

#THIRD -> DATA IS CLEANED:
#sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
# AVERAGE AGE FOR CLASS 1: 37, CLASS 2:29, CLASS 3:24

def impute_age(cols):

    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24
    else:
        
        return Age

#Function is applied to the Age in order to impute for null values
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)

#sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

train.drop('Cabin',axis=1, inplace = True)

#FORTH -> CATEGORICAL FEATURES ARE CONVERTED TO DUMMY VARIABLES USING PANDAS

sex = pd.get_dummies(train['Sex'],drop_first = True)
embark = pd.get_dummies(train['Embarked'],drop_first = True)
train.drop(['Sex','Embarked','Name','Ticket'],axis = 1, inplace = True)
train = pd.concat([train,sex,embark],axis = 1)
#print(train.columns)

#same procedure is done for the testing data:

test=pd.read_csv('test.csv')
test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)
test.drop('Cabin',axis = 1,inplace = True)
sex_test = pd.get_dummies(test['Sex'],drop_first = True)
embark_test = pd.get_dummies(test['Embarked'],drop_first = True)
test.drop(['Sex','Embarked','Name','Ticket'],axis = 1,inplace = True)

X_test = pd.concat([test,sex_test,embark_test],axis = 1)
X_test = X_test.fillna(0)      #   *** X_test ****

#print(np.any(np.isnan(X_test)))
#print(X_test.columns)


y_test = pd.read_csv('gender_submission.csv')

y_test = y_test['Survived']   #    *** y_test ****


#print(y_test.columns)


#SIXTH: LOGISITC MODEL IS BUILT AND DATA IS TRAINED ON IT
from sklearn.linear_model import LogisticRegression

X_train = train.drop('Survived',axis=1)     #   *** X_test ****

y_train = train['Survived']                 #   *** y_train ****


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
#print(logmodel)

#SEVENTH: PREDICTED Y VALUES FOR THE TEST SET IS CALCULATED
predicted_y = logmodel.predict(X_test)
predictions = pd.DataFrame(predicted_y,index = X_test['PassengerId'],columns = ['Survived']) 
print(len(predicted_y))
predictions.to_csv('model_prediction.csv')
