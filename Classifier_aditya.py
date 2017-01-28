import pandas as pd
from team_stats import *
from plots import *
from sklearn import svm
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

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
score_pred_log = {}

for team in teams:
	score_true[team] = 0
	score_pred_svm[team] = 0
	score_pred_lin[team] = 0
	score_pred_rf[team] = 0
	score_pred_log[team] = 0


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

#-------------------SVM --------------------------
clf = svm.SVC(kernel='linear', C=1).fit(X_train_data, X_train_target)
X_predicted = clf.predict(X_test_data)
print('SVM ',metrics.accuracy_score(X_test_target, X_predicted))
# Compute confusion matrix
cnf_matrix = confusion_matrix(X_test_target, X_predicted)
# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix,class_names,plt.cm.Blues,title='Confusion matrix for SVM')
#plot the histogram for the teams
plot_win_stats(dataframe1, X_predicted, 'SVM_team_win_stats')



for index in range(len(X_predicted)):
	if X_predicted[index] == 1:
		score_pred_svm[hometeam[index]] += 3
	elif X_predicted[index] == 3:
		score_pred_svm[hometeam[index]] += 1
		score_pred_svm[awayteam[index]] += 1
	else:
		score_pred_svm[awayteam[index]] += 3

#-----------------------Linear SVC----------------------
linear = LinearSVC().fit(X_train_data, X_train_target)
X_predicted_lsvc = linear.predict(X_test_data)
print('linearSVC ',metrics.accuracy_score(X_test_target, X_predicted_lsvc))
# Compute confusion matrix
cnf_matrix = confusion_matrix(X_test_target, X_predicted_lsvc)
# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix,class_names,plt.cm.Blues,title='Confusion matrix for LinearSVC')
#plot the histogram for the teams
plot_win_stats(dataframe1, X_predicted_lsvc, 'LinearSVC_team_win_stats')


for index in range(len(X_predicted_lsvc)):
	if X_predicted_lsvc[index] == 1:
		score_pred_lin[hometeam[index]] += 3
	elif X_predicted_lsvc[index] == 3:
		score_pred_lin[hometeam[index]] += 1
		score_pred_lin[awayteam[index]] += 1
	else:
		score_pred_lin[awayteam[index]] += 3

#-------------------------Random Forest--------------------
clf = RandomForestClassifier().fit(X_train_data, X_train_target)
X_predicted_rf = clf.predict(X_test_data)
print('RF ',metrics.accuracy_score(X_test_target, X_predicted_rf))
# Compute confusion matrix
cnf_matrix = confusion_matrix(X_test_target, X_predicted_rf)
# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix,class_names,plt.cm.Blues,title='Confusion matrix for RandomForest')
#plot the histogram for the teams
plot_win_stats(dataframe1, X_predicted_rf, 'RF_team_win_stats')


for index in range(len(X_predicted_rf)):
	if X_predicted_rf[index] == 1:
		score_pred_rf[hometeam[index]] += 3
	elif X_predicted_rf[index] == 3:
		score_pred_rf[hometeam[index]] += 1
		score_pred_rf[awayteam[index]] += 1
	else:
		score_pred_rf[awayteam[index]] += 3


#-----------------------Logistic Regression----------------------
log = LogisticRegression().fit(X_train_data, X_train_target)
X_predicted_log = log.predict(X_test_data)
print('logistic Regression ',metrics.accuracy_score(X_test_target, X_predicted_log))
# Compute confusion matrix
cnf_matrix = confusion_matrix(X_test_target, X_predicted_log)
# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix,class_names,plt.cm.Blues,title='Confusion matrix for Logistic Regression')
#plot the histogram for the teams
plot_win_stats(dataframe1, X_predicted_log, 'LogRegression_team_win_stats')



for index in range(len(X_predicted_log)):
	if X_predicted_log[index] == 1:
		score_pred_log[hometeam[index]] += 3
	elif X_predicted_log[index] == 3:
		score_pred_log[hometeam[index]] += 1
		score_pred_log[awayteam[index]] += 1
	else:
		score_pred_log[awayteam[index]] += 3


score_true = sorted(score_true,key=score_true.get,reverse=True)[:10]
score_pred_svm = sorted(score_pred_svm,key=score_pred_svm.get,reverse=True)[:10]
score_pred_lin = sorted(score_pred_lin,key=score_pred_lin.get,reverse=True)[:10]
score_pred_rf = sorted(score_pred_rf,key=score_pred_rf.get,reverse=True)[:10]
score_pred_log = sorted(score_pred_log,key=score_pred_log.get,reverse=True)[:10]

print( 'svm ',len(set(score_true) & set(score_pred_svm)))
print( 'linsvc ',len(set(score_true) & set(score_pred_lin)))
print( 'rf ',len(set(score_true) & set(score_pred_rf)))
print( 'lr ',len(set(score_true) & set(score_pred_log)))