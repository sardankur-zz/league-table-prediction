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
dataframe1 = pd.read_pickle('dataframeTestData.p')

X_train_data = []
X_train_target = []
X_test_data = []
X_test_target = []


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

#---------------------- Test Data -------------------------------------

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
score_true = {}
score_pred_svm = {}
score_pred_lin = {}
score_pred_rf = {}
score_pred_lr = {}

for team in teams:
	score_true[team] = 0
	score_pred_svm[team] = 0
	score_pred_lin[team] = 0
	score_pred_rf[team] = 0
	score_pred_lr[team] = 0


for item in list(dataframe1['FTR']):
	if item == 'H': #home
		X_test_target.append(1)
	elif item == 'A': #away
		X_test_target.append(2)
	else: #draw
		X_test_target.append(3)


for index in range(len(dataframe1)):
	X_test_data.append([rating_home[index],rating_away[index],_5HW[index],_5HL[index],_5HD[index],_5AW[index],_5AL[index],_5AD[index],_5FTHGS[index],_5FTHGC[index],_5FTAGS[index],_5FTAGC[index],_5HST[index],_5AST[index],_5HC[index],_5AC[index],_5HR[index],_5AR[index],_H2H5FTHGS[index],_H2H5FTAGS[index],_H2H5FTHGC[index],_H2H5FTAGC[index],_H2H5HST[index],_H2H5AST[index],_H2H5HC[index],_H2H5AC[index],_H2H5HR[index],_H2H5AR[index]])
	if X_test_target[index] == 1:
		score_true[hometeam[index]] += 3
	elif X_test_target[index] == 3:
		score_true[awayteam[index]] += 1
		score_true[hometeam[index]] += 1
	else:
		score_true[awayteam[index]] += 3

#order teams by their score from highest to lowest
score_true = sorted(score_true,key=score_true.get,reverse=True)

'''
#------------------------------------SVM--------------------------------------
#{'C': [1,10,100], 'kernel':['linear'], 'class_weight':['balanced', None]},
param_dist = [{'C': [1,10,100], 'gamma': [0.001,0.0001], 'kernel':['rbf'], 'class_weight':['balanced', None]}]
grid_search = GridSearchCV(svm.SVC(), param_dist, cv=5)
grid_search.fit(X_train_data, X_train_target)

X_predicted_svm = grid_search.predict(X_test_data)

#svr = svm.SVC(kernel='linear', C=1).fit(X_train_data, X_train_target)
#X_predicted_svm = svr.predict(X_test_data)

cnf_matrix = confusion_matrix(X_test_target, X_predicted_svm)
# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix,class_names,plt.cm.Blues,title='Confusion matrix for SVM')
plot_win_stats(dataframe1, X_predicted_svm, 'SVM_team_win_stats')
print('SVM:accuracy: ',metrics.accuracy_score(X_test_target, X_predicted_svm))
for index in range(len(X_predicted_svm)):
	if X_predicted_svm[index] == 1:
		score_pred_svm[hometeam[index]] += 3
	elif X_predicted_svm[index] == 3:
		score_pred_svm[hometeam[index]] += 1
		score_pred_svm[awayteam[index]] += 1
	else:
		score_pred_svm[awayteam[index]] += 3


score_pred_svm_top_5 = sorted(score_pred_svm,key=score_pred_svm.get,reverse=True)[:5]
score_pred_svm_top_10 = sorted(score_pred_svm,key=score_pred_svm.get,reverse=True)[:10]
score_pred_svm_top_15 = sorted(score_pred_svm,key=score_pred_svm.get,reverse=True)[:15]

print( 'SVM:common in top 5: ',len(set(score_true[:5]) & set(score_pred_svm_top_5)))
print( 'SVM:common in top 10: ',len(set(score_true[:10]) & set(score_pred_svm_top_10)))
print( 'SVM:common in top 15: ',len(set(score_true[:15]) & set(score_pred_svm_top_15)))
'''
#---------------------------RANDOM FOREST--------------------------------------

param_dist = {"max_depth": [3,7,None],"max_features": [4,8,11,15,18],"min_samples_split": [4,7,11],"min_samples_leaf": [4,7,11],"bootstrap": [True, False],"criterion": ["gini", "entropy"]}

grid_search = GridSearchCV(RandomForestClassifier(n_estimators=20), param_dist, cv=5)
grid_search.fit(X_train_data, X_train_target)

X_predicted_rf = grid_search.predict(X_test_data)
'''

rfc = RandomForestClassifier().fit(X_train_data, X_train_target)
X_predicted_rf = rfc.predict(X_test_data)
'''
cnf_matrix = confusion_matrix(X_test_target, X_predicted_rf)
# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix,class_names,plt.cm.Blues,title='Confusion matrix for Random Forest')
plot_win_stats(dataframe1, X_predicted_rf, 'RandomForest_team_win_stats')
print('RF:accuracy: ',metrics.accuracy_score(X_test_target, X_predicted_rf))

for index in range(len(X_predicted_rf)):
	if X_predicted_rf[index] == 1:
		score_pred_rf[hometeam[index]] += 3
	elif X_predicted_rf[index] == 3:
		score_pred_rf[hometeam[index]] += 1
		score_pred_rf[awayteam[index]] += 1
	else:
		score_pred_rf[awayteam[index]] += 3


score_pred_rf_top_5 = sorted(score_pred_rf,key=score_pred_rf.get,reverse=True)[:5]
score_pred_rf_top_10 = sorted(score_pred_rf,key=score_pred_rf.get,reverse=True)[:10]
score_pred_rf_top_15 = sorted(score_pred_rf,key=score_pred_rf.get,reverse=True)[:15]

print( 'RF:common in top 5: ',len(set(score_true[:5]) & set(score_pred_rf_top_5)))
print( 'RF:common in top 10: ',len(set(score_true[:10]) & set(score_pred_rf_top_10)))
print( 'RF:common in top 15: ',len(set(score_true[:15]) & set(score_pred_rf_top_15)))
'''
#---------------------------LOGISTIC REGRESSION--------------------------------

param_dist = {"solver" : ['newton-cg', 'lbfgs', 'liblinear', 'sag'],'C': [1,10,100,1000],'class_weight':['balanced', None], 'max_iter':[1000]}
grid_search = GridSearchCV(LogisticRegression(), param_dist, cv=5)
grid_search.fit(X_train_data, X_train_target)

X_predicted_lr = grid_search.predict(X_test_data)

#lrc = LogisticRegression().fit(X_train_data, X_train_target)
#X_predicted_lr = lrc.predict(X_test_data)

cnf_matrix = confusion_matrix(X_test_target, X_predicted_lr)
# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix,class_names,plt.cm.Blues,title='Confusion matrix for Logistic Regression')
plot_win_stats(dataframe1, X_predicted_lr, 'LogisticRegression_team_win_stats')

print('LR:accuracy: ',metrics.accuracy_score(X_test_target, X_predicted_lr))

for index in range(len(X_predicted_lr)):
	if X_predicted_lr[index] == 1:
		score_pred_lr[hometeam[index]] += 3
	elif X_predicted_lr[index] == 3:
		score_pred_lr[hometeam[index]] += 1
		score_pred_lr[awayteam[index]] += 1
	else:
		score_pred_lr[awayteam[index]] += 3

score_pred_lr_top_5 = sorted(score_pred_lr,key=score_pred_lr.get,reverse=True)[:5]
score_pred_lr_top_10 = sorted(score_pred_lr,key=score_pred_lr.get,reverse=True)[:10]
score_pred_lr_top_15 = sorted(score_pred_lr,key=score_pred_lr.get,reverse=True)[:15]

print( 'LR:common in top 5: ',len(set(score_true[:5]) & set(score_pred_lr_top_5)))
print( 'LR:common in top 10: ',len(set(score_true[:10]) & set(score_pred_lr_top_10)))
print( 'LR:common in top 15: ',len(set(score_true[:15]) & set(score_pred_lr_top_15)))
'''
'''
#-----------------------Linear SVC----------------------

param_dist = {'C': [1,10,100,1000],'class_weight':['balanced', None]}
grid_search = GridSearchCV(LinearSVC(), param_dist, cv=5)
grid_search.fit(X_train_data, X_train_target)
X_predicted_lsvc = grid_search.predict(X_test_data)


#linear = LinearSVC().fit(X_train_data, X_train_target)
#X_predicted_lsvc = linear.predict(X_test_data)

# Compute confusion matrix
cnf_matrix = confusion_matrix(X_test_target, X_predicted_lsvc)
# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix,class_names,plt.cm.Blues,title='Confusion matrix for LinearSVC')
plot_win_stats(dataframe1, X_predicted_lsvc, 'LinSVC_team_win_stats')

print('linearSVC:accuracy: ',metrics.accuracy_score(X_test_target, X_predicted_lsvc))

for index in range(len(X_predicted_lsvc)):
	if X_predicted_lsvc[index] == 1:
		score_pred_lin[hometeam[index]] += 3
	elif X_predicted_lsvc[index] == 3:
		score_pred_lin[hometeam[index]] += 1
		score_pred_lin[awayteam[index]] += 1
	else:
		score_pred_lin[awayteam[index]] += 3


score_pred_lin_top_5 = sorted(score_pred_lin,key=score_pred_lin.get,reverse=True)[:5]
score_pred_lin_top_10 = sorted(score_pred_lin,key=score_pred_lin.get,reverse=True)[:10]
score_pred_lin_top_15 = sorted(score_pred_lin,key=score_pred_lin.get,reverse=True)[:15]

print( 'linearSVC:common in top 5: ',len(set(score_true[:5]) & set(score_pred_lin_top_5)))
print( 'linearSVC:common in top 10: ',len(set(score_true[:10]) & set(score_pred_lin_top_10)))
print( 'linearSVC:common in top 15: ',len(set(score_true[:15]) & set(score_pred_lin_top_15)))
'''
