import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn import metrics
import numbers
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


dataframe = pd.read_pickle('dataframe.p')
dataframe1 = pd.read_pickle('dataframeTestData.p')

X_train_data = []
X_train_target = []
X_test_data = []
X_test_target = []

feature_position = {}
feature_position['rating_home'] = 0
feature_position['rating_away'] = 1
feature_position['5HW'] = 2
feature_position['5HL'] = 3
feature_position['5HD'] = 4
feature_position['5AW'] = 5
feature_position['5AL'] =  6
feature_position['5AD'] = 7
feature_position['5FTHGS'] = 8
feature_position['5FTHGC'] = 9
feature_position['5FTAGS'] = 10
feature_position['5FTAGC'] = 11
feature_position['5HST'] = 12
feature_position['5AST'] = 13
feature_position['5HC'] = 14
feature_position['5AC'] = 15
feature_position['5HR'] = 16
feature_position['5AR'] = 17
feature_position['H2H5FTHGS'] = 18
feature_position['H2H5FTAGS'] = 19
feature_position['H2H5FTHGC'] = 20
feature_position['H2H5FTAGC'] = 21
feature_position['H2H5HST'] = 22
feature_position['H2H5AST'] = 23
feature_position['H2H5HC'] = 24
feature_position['H2H5AC'] = 25
feature_position['H2H5HR'] = 26
feature_position['H2H5AR'] = 27



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



#print(len(dataframe))
for item in list(dataframe['FTR']):
	if item == 'H': #home
		X_train_target.append(1)
	elif item == 'A': #away
		X_train_target.append(2)
	else: #draw
		X_train_target.append(3)


for index in range(len(dataframe)):
	X_train_data.append([rating_home[index],rating_away[index],_5HW[index],_5HL[index],_5HD[index],_5AW[index],_5AL[index],_5AD[index],_5FTHGS[index],_5FTHGC[index],_5FTAGS[index],_5FTAGC[index],_5HST[index],_5AST[index],_5HC[index],_5AC[index],_5HR[index],_5AR[index],_H2H5FTHGS[index],_H2H5FTAGS[index],_H2H5FTHGC[index],_H2H5FTAGC[index],_H2H5HST[index],_H2H5AST[index],_H2H5HC[index],_H2H5AC[index],_H2H5HR[index],_H2H5AR[index]])





#-----------------------------------------------------------

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

for team in teams:
	score_true[team] = 0

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
		score_true[hometeam[index]] += 2
	elif X_test_target[index] == 3:
		score_true[awayteam[index]] += 1
		score_true[awayteam[index]] += 1

#---------------------------------------------

def feature_select(features, data):
    X = []
    for row in data:
        newrow = []
        for f in features:
            newrow.append(row[feature_position[f]])
        X.append(newrow)
    return X

score_true = sorted(score_true,key=score_true.get,reverse=True)[:10]

def select_features(features):

    X_selected_train = feature_select(features, X_train_data)
    X_selected_test = feature_select(features, X_test_data)

    score_pred_svm = {}
    score_pred_lin = {}
    score_pred_rf = {}
    score_pred_lrc = {}

    accuracy_svm = 0
    accuracy_linear_svm = 0
    accuracy_linear_rf = 0

    for team in teams:
        score_pred_svm[team] = 0
        score_pred_lin[team] = 0
        score_pred_rf[team] = 0
        score_pred_lrc[team] = 0

    clf = svm.SVC(kernel='linear', C=1).fit(X_selected_train, X_train_target)
    X_predicted = clf.predict(X_selected_test)

    accuracy_svm = metrics.accuracy_score(X_test_target, X_predicted)
    print('SVM ', accuracy_svm)

    for index in range(len(X_predicted)):
        if X_predicted[index] == 1:
            score_pred_svm[hometeam[index]] += 2
        elif X_predicted[index] == 3:
            score_pred_svm[hometeam[index]] += 1
            score_pred_svm[awayteam[index]] += 1

    #---------------------------------------------
    linear = LinearSVC(C=100, class_weight='balanced').fit(X_selected_train, X_train_target)
    X_predicted_lsvc = linear.predict(X_selected_test)

    accuracy_linear_svm = metrics.accuracy_score(X_test_target, X_predicted_lsvc)
    print('linearSVC ', accuracy_linear_svm)

    for index in range(len(X_predicted_lsvc)):
        if X_predicted_lsvc[index] == 1:
            score_pred_lin[hometeam[index]] += 2
        elif X_predicted_lsvc[index] == 3:
            score_pred_lin[hometeam[index]] += 1
            score_pred_lin[awayteam[index]] += 1

    #---------------------------------------------
    clf = RandomForestClassifier(min_samples_leaf=1, criterion='gini', min_samples_split = 7, max_depth = 7,
                                bootstrap = False).fit(X_selected_train, X_train_target)
    X_predicted_rf = clf.predict(X_selected_test)

    accuracy_linear_rf = metrics.accuracy_score(X_test_target, X_predicted_rf)
    print('RF ',accuracy_linear_rf)

    for index in range(len(X_predicted_rf)):
        if X_predicted_rf[index] == 1:
            score_pred_rf[hometeam[index]] += 2
        elif X_predicted_rf[index] == 3:
            score_pred_rf[hometeam[index]] += 1
            score_pred_rf[awayteam[index]] += 1

    # ---------------------------------------------
    lrc = LogisticRegression(solver='liblinear', max_iter=1000, C = 100, class_weight=None).fit(X_selected_train, X_train_target)
    X_predicted_lr = lrc.predict(X_selected_test)

    accuracy_linear_lrc = metrics.accuracy_score(X_test_target, X_predicted_lr)
    print('LR ', accuracy_linear_lrc)

    for index in range(len(X_predicted_lr)):
        if X_predicted_lr[index] == 1:
            score_pred_lrc[hometeam[index]] += 2
        elif X_predicted_lr[index] == 3:
            score_pred_lrc[hometeam[index]] += 1
            score_pred_lrc[awayteam[index]] += 1

    score_pred_svm = sorted(score_pred_svm,key=score_pred_svm.get,reverse=True)[:10]
    score_pred_lin = sorted(score_pred_lin,key=score_pred_lin.get,reverse=True)[:10]
    score_pred_rf = sorted(score_pred_rf,key=score_pred_rf.get,reverse=True)[:10]
    score_pred_lrc = sorted(score_pred_lrc, key=score_pred_lrc.get, reverse=True)[:10]

    set_svm = set(score_true) & set(score_pred_svm)
    set_linear_svm = set(score_true) & set(score_pred_lin)
    set_rf =  set(score_true) & set(score_pred_rf)
    set_lrc = set(score_true) & set(score_pred_lrc)

    print( 'svm ',len(set_svm))
    print( 'linsvc ',len(set_linear_svm))
    print( 'rf ', len(set_rf))
    print('lrc ', len(set_lrc))

    return [accuracy_svm, accuracy_linear_svm, accuracy_linear_rf, accuracy_linear_lrc,
            len(set_svm), len(set_linear_svm), len(set_rf), len(set_lrc)]






