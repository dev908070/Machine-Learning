import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


## step 1:- get the data
dataset = pd.read_csv("data_pre.csv")

#feature matric 
X = dataset.iloc[:,0:3].values
# vector of prediction
y = dataset.iloc[:,-1].values

## step 2 :-data visualization
## step 3:- Data preprocessing
#handling missing values

from sklearn.impute import SimpleImputer
sim = SimpleImputer(missing_values = np.nan, strategy= "median")
sim.fit(X[:,0:2])
X[:,0:2]= sim.transform( X[:,0:2])
sim.statistics_

# fit(X) does the computation only
#transform(X) Return a copy

from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
X[:,2] = lab.fit_transform(X[:,2])
y = lab.fit_transform(y)
lab.classes_

#label encoding is use for converting the catagorical(string ) to integer values
# after thata there is a problem occurs which is called dummy variable trap
# for this problem we can use dumy variable encoding (sparce matrix)


dataset1 = pd.get_dummies(dataset)


##### ADULT SALARY DATASET##########

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#  step 1 :- get the data

dataset = pd.read_csv("adult_salary.csv")
 
# there is a problem in this dataset column name are missing
#so we have to give the name according to data description

col_names = ['age','workclass','fnlwgt','education','eduction-num',
             'marital-status','occupation','relationship','race','gender',
             'capital-gain','capital-loss','hours-per-week',
             'native-country','salary']


dataset = pd.read_csv("adult_salary.csv",names=col_names,na_values=' ?')



dataset.isnull()
#is null retun true or false so we have to use sum() method to calculate the nan vales

dataset.isnull().sum()
# after na_values = x we can count na values
# its shows nan is not present in our dataset but its not true

 """ there is a majaor problem calculating or fing the null values in dataset
    because the nan value is marked with diff. character or letter 
    for that we have to read description of dataset"""

 
# a good practice is that to separate feature mmatrix or vector of predection

#numerical =[0,2,4,10,11,12]
#categorical = [1,3,5,6,7,8,9,13,14]


X = dataset.iloc[:,0:14]
y = dataset.iloc[:,-1]


# best practice is that to not to tough the origenal data rather you can create copy of dataset

## filling the nan values with the help of pandas 
temp = X[['workclass','occupation','native-country']]
temp['workclass'].value_counts()
temp['occupation'].value_counts()
temp['native-country'].value_counts()

temp['workclass'] = temp['workclass'].fillna(' private')

temp['occupation'] = temp['workclass'].fillna(' Prof-specialty')

temp['native-country'] = temp['workclass'].fillna(' United-States')

temp.isnull().sum()



# now fill the missing values with the help of sklearn library
from sklearn.impute import SimpleImputer

sim=SimpleImputer(strategy='most_frequent')
sim.fit(temp)
temp = sim.transform(temp)

temp.isnull().sum()


X[['workclass','occupation','native-country']]= temp

X.isnull().sum()

# Label encoding is use for converting the catgorical columns into integer

X = pd.get_dummies(X)
#label encoding with sklearn

from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
lab.fit_transform(y) 
lab.classes_

## feature scalling(it means convert all the data in similar scale as like km to meter kg to gram)
from sklearn.preprocessing import MinMaxScaler,StandardScaler
min_max = MinMaxScaler()
sc = StandardScaler()

X_min_max = min_max.fit_transform(X)
X_sc = sc.fit_transform(X)


