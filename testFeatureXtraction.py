import pandas as pd
from featureCalculation import *

season = ['15-16']
dataframe = pd.DataFrame()
list_ = []
for s in season:
    df = pd.read_csv('epl_' + s + '.txt')
    col = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
           'Referee', 'HST', 'AST', 'HC', 'AC', 'HR', 'AR', 'rating_home', 'rating_away']
    df_temp = df[col]
    df_temp = df_temp
    list_.append(df_temp)
dataframe = pd.concat(list_)
dataframe = dataframe.dropna()



# Calculating new features and adding those in the new dataframe 
#extra features extracted out of the match data
_5HW = []
_5HL = []
_5HD = []
_5AW = []
_5AL = []
_5AD = []
_5FTHGS = []
_5FTHGC = []
_5FTAGS = []
_5FTAGC = []
_5HST = []
_5AST = []
_5HC = []
_5AC = []
_5HR = []
_5AR = []
_H2H5FTHGS = []
_H2H5FTAGS = []
_H2H5FTHGC = []
_H2H5FTAGC = []
_H2H5HST = []
_H2H5AST = []
_H2H5HC = []
_H2H5AC = []
_H2H5HR = []
_H2H5AR = []

# building the final data frame
for _,row in dataframe.iterrows():
    _5HW.append(get_number_of_wins(dataframe, str(row['Date']), str(row['HomeTeam'])))
    _5AW.append(get_number_of_wins(dataframe, str(row['Date']), str(row['AwayTeam'])))
    _5HD.append(get_number_of_draws(dataframe, str(row['Date']), str(row['HomeTeam'])))
    _5AD.append(get_number_of_draws(dataframe, str(row['Date']), str(row['AwayTeam'])))
    _5HL.append(get_number_of_losses(dataframe, str(row['Date']), str(row['HomeTeam'])))
    _5AL.append(get_number_of_losses(dataframe, str(row['Date']), str(row['AwayTeam'])))
    _5FTHGS.append(get_number_of_goals_scored(dataframe, str(row['Date']), str(row['HomeTeam'])))
    _5FTHGC.append(get_number_of_goals_conceded(dataframe, str(row['Date']), str(row['HomeTeam'])))
    _5FTAGS.append(get_number_of_goals_scored(dataframe, str(row['Date']), str(row['AwayTeam'])))
    _5FTAGC.append(get_number_of_goals_conceded(dataframe, str(row['Date']), str(row['AwayTeam'])))
    _5HST.append(get_number_of_shots_on_target(dataframe, str(row['Date']), str(row['HomeTeam'])))
    _5AST.append(get_number_of_shots_on_target(dataframe, str(row['Date']), str(row['AwayTeam'])))
    _5HC.append(get_number_of_corners(dataframe, str(row['Date']), str(row['HomeTeam'])))
    _5AC.append(get_number_of_corners(dataframe, str(row['Date']), str(row['AwayTeam'])))
    _5HR.append(get_number_of_red_cards(dataframe, str(row['Date']), str(row['HomeTeam'])))
    _5AR.append(get_number_of_red_cards(dataframe, str(row['Date']), str(row['AwayTeam'])))
    _H2H5FTHGS.append(get_number_of_h2h_home_goals_scored(dataframe, str(row['Date']), str(row['HomeTeam']), str(row['AwayTeam']), str(row['HomeTeam'])))
    _H2H5FTAGS.append(get_number_of_h2h_away_goals_scored(dataframe, str(row['Date']), str(row['HomeTeam']), str(row['AwayTeam']), str(row['AwayTeam'])))
    _H2H5FTHGC.append(get_number_of_h2h_home_goals_conceded(dataframe, str(row['Date']), str(row['HomeTeam']), str(row['AwayTeam']), str(row['HomeTeam'])))
    _H2H5FTAGC.append(get_number_of_h2h_away_goals_conceded(dataframe, str(row['Date']), str(row['HomeTeam']), str(row['AwayTeam']), str(row['AwayTeam'])))
    _H2H5HST.append(get_number_of_h2h_home_shot_on_target(dataframe, str(row['Date']), str(row['HomeTeam']), str(row['AwayTeam']), str(row['HomeTeam'])))
    _H2H5AST.append(get_number_of_h2h_away_shot_on_target(dataframe, str(row['Date']), str(row['HomeTeam']), str(row['AwayTeam']), str(row['AwayTeam'])))
    _H2H5HC.append(get_number_of_h2h_home_corners(dataframe, str(row['Date']), str(row['HomeTeam']), str(row['AwayTeam']), str(row['HomeTeam'])))
    _H2H5AC.append(get_number_of_h2h_away_corners(dataframe, str(row['Date']), str(row['HomeTeam']), str(row['AwayTeam']), str(row['AwayTeam'])))
    _H2H5HR.append(get_number_of_h2h_home_red_cards(dataframe, str(row['Date']), str(row['HomeTeam']), str(row['AwayTeam']), str(row['HomeTeam'])))
    _H2H5AR.append(get_number_of_h2h_away_red_cards(dataframe, str(row['Date']), str(row['HomeTeam']), str(row['AwayTeam']), str(row['AwayTeam'])))
dataframe['5HW'] = _5HW
dataframe['5AW'] = _5AW
dataframe['5HD'] = _5HD
dataframe['5AD'] = _5AD
dataframe['5HL'] = _5HL
dataframe['5AL'] = _5AL
dataframe['5FTHGS'] = _5FTHGS
dataframe['5FTHGC'] = _5FTHGC
dataframe['5FTAGS'] = _5FTAGS
dataframe['5FTAGC'] = _5FTAGC
dataframe['5HST']=_5HST
dataframe['5AST']=_5AST
dataframe['5HC']=_5HC
dataframe['5AC']=_5AC
dataframe['5HR'] = _5HR
dataframe['5AR']=_5AR
dataframe['H2H5FTHGS']=_H2H5FTHGS
dataframe['H2H5FTAGS']=_H2H5FTAGS
dataframe['H2H5FTHGC']=_H2H5FTHGC
dataframe['H2H5FTAGC']=_H2H5FTAGC
dataframe['H2H5HST']=_H2H5HST
dataframe['H2H5AST']=_H2H5AST
dataframe['H2H5HC']=_H2H5HC
dataframe['H2H5AC']=_H2H5AC
dataframe['H2H5HR']=_H2H5HR
dataframe['H2H5AR']=_H2H5AR


dataframe = dataframe.dropna()

print("done")

dataframe.to_pickle('dataframeTestData.p')



