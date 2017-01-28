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

dataframe1 = pd.read_pickle('dataframeTestData.p')
X_test_data = []
X_test_target = []
hometeam = list(dataframe1['HomeTeam'])
awayteam = list(dataframe1['AwayTeam'])
rating_home = list(dataframe1['rating_home'])
rating_away = list(dataframe1['rating_away'])
_5HW = list(dataframe1['5HW'])
_5HL = list(dataframe1['5HL'])
_5HD = list(dataframe1['5HD'])
_5AW = list(dataframe1['5AW'])
_5AL = list(dataframe1['5AL'])
_5AD = list(dataframe1['5AD'])
_5FTHGS = list(dataframe1['5FTHGS'])
_5FTHGC = list(dataframe1['5FTHGC'])
_5FTAGS = list(dataframe1['5FTAGS'])
_5FTAGC = list(dataframe1['5FTAGC'])
_5HST = list(dataframe1['5HST'])
_5AST = list(dataframe1['5AST'])
_5HC = list(dataframe1['5HC'])
_5AC = list(dataframe1['5AC'])
_5HR = list(dataframe1['5HR'])
_5AR = list(dataframe1['5AR'])
_H2H5FTHGS = list(dataframe1['H2H5FTHGS'])
_H2H5FTAGS = list(dataframe1['H2H5FTAGS'])
_H2H5FTHGC = list(dataframe1['H2H5FTHGC'])
_H2H5FTAGC = list(dataframe1['H2H5FTAGC'])
_H2H5HST = list(dataframe1['H2H5HST'])
_H2H5AST = list(dataframe1['H2H5AST'])
_H2H5HC = list(dataframe1['H2H5HC'])
_H2H5AC = list(dataframe1['H2H5AC'])
_H2H5HR = list(dataframe1['H2H5HR'])
_H2H5AR = list(dataframe1['H2H5AR'])
teams = list(set(dataframe1['HomeTeam']) | set(dataframe1['AwayTeam']))
for item in list(dataframe1['FTR']):
	if item == 'H': #home
		X_test_target.append(1)
	elif item == 'A': #away
		X_test_target.append(2)
	else: #draw
		X_test_target.append(3)


for index in range(len(dataframe1)):
	X_test_data.append([rating_home[index],rating_away[index],_5HW[index],_5HL[index],_5HD[index],_5AW[index],_5AL[index],_5AD[index],_5FTHGS[index],_5FTHGC[index],_5FTAGS[index],_5FTAGC[index],_5HST[index],_5AST[index],_5HC[index],_5AC[index],_5HR[index],_5AR[index],_H2H5FTHGS[index],_H2H5FTAGS[index],_H2H5FTHGC[index],_H2H5FTAGC[index],_H2H5HST[index],_H2H5AST[index],_H2H5HC[index],_H2H5AC[index],_H2H5HR[index],_H2H5AR[index]])



classifier_object = joblib.load('BestClassifier.pkl')
#above classifier can now be used on any test data as follows:
X_predicted = classifier_object.predict(X_test_data)

print(metrics.accuracy_score(X_test_target, X_predicted))

