from sklearn.externals import joblib
import pandas as pd
from plots import *
from sklearn import svm
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import numpy
from team_stats import *
import scipy
from sklearn.model_selection import *
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

class_names = ['Win', 'Loss', 'Draw']

dataframe = pd.read_pickle('dataframe.p')

X_train_data = []
X_train_target = []


#------------ Train Data -----------------------
rating_home = list(dataframe['rating_home'])
rating_away = list(dataframe['rating_away'])
_5HW = list(dataframe['5HW'])
_5HL = list(dataframe['5HL'])
_5HD = list(dataframe['5HD'])
_5AW = list(dataframe['5AW'])
_5AL = list(dataframe['5AL'])
_5AD = list(dataframe['5AD'])
_5FTHGS = list(dataframe['5FTHGS'])
_5FTHGC = list(dataframe['5FTHGC'])
_5FTAGS = list(dataframe['5FTAGS'])
_5FTAGC = list(dataframe['5FTAGC'])
_5HST = list(dataframe['5HST'])
_5AST = list(dataframe['5AST'])
_5HC = list(dataframe['5HC'])
_5AC = list(dataframe['5AC'])
_5HR = list(dataframe['5HR'])
_5AR = list(dataframe['5AR'])
_H2H5FTHGS = list(dataframe['H2H5FTHGS'])
_H2H5FTAGS = list(dataframe['H2H5FTAGS'])
_H2H5FTHGC = list(dataframe['H2H5FTHGC'])
_H2H5FTAGC = list(dataframe['H2H5FTAGC'])
_H2H5HST = list(dataframe['H2H5HST'])
_H2H5AST = list(dataframe['H2H5AST'])
_H2H5HC = list(dataframe['H2H5HC'])
_H2H5AC = list(dataframe['H2H5AC'])
_H2H5HR = list(dataframe['H2H5HR'])
_H2H5AR = list(dataframe['H2H5AR'])


for item in list(dataframe['FTR']):
	if item == 'H': #home
		X_train_target.append(1)
	elif item == 'A': #away
		X_train_target.append(2)
	else: #draw
		X_train_target.append(3)


for index in range(len(dataframe)):
	X_train_data.append([rating_home[index],rating_away[index],_5HW[index],_5HL[index],_5HD[index],_5AW[index],_5AL[index],_5AD[index],_5FTHGS[index],_5FTHGC[index],_5FTAGS[index],_5FTAGC[index],_5HST[index],_5AST[index],_5HC[index],_5AC[index],_5HR[index],_5AR[index],_H2H5FTHGS[index],_H2H5FTAGS[index],_H2H5FTHGC[index],_H2H5FTAGC[index],_H2H5HST[index],_H2H5AST[index],_H2H5HC[index],_H2H5AC[index],_H2H5HR[index],_H2H5AR[index]])



#---------------------------RANDOM FOREST--------------------------------------
'''
param_dist = {"max_depth": [3,7,None],"max_features": [4,8,11,15,18],"min_samples_split": [4,7,11],"min_samples_leaf": [4,7,11],"bootstrap": [True, False],"criterion": ["gini", "entropy"]}

grid_search = GridSearchCV(RandomForestClassifier(n_estimators=20), param_dist, cv=5)
grid_search.fit(X_train_data, X_train_target)
print('random forest best config ',grid_search.best_params_)
X_predicted_rf = grid_search.predict(X_test_data)
'''

rfc = RandomForestClassifier(min_samples_leaf= 7, criterion= 'gini', min_samples_split= 7, max_depth= 7, max_features= 11, bootstrap= False).fit(X_train_data, X_train_target)
joblib.dump(rfc,'BestClassifier.pkl')

